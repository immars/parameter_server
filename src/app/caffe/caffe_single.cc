#include <iostream>
#include <unistd.h>
#include <chrono>
#include <thread>
#include <google/protobuf/io/coded_stream.h>
#include <sys/time.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "caffe/caffe.hpp"
#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/upgrade_proto.hpp"
#include "app/caffe/util.h"
#include "app/caffe/forwarder_solver.h"
#include "ps.h"

using namespace caffe;
using namespace std;
using caffe::Blob;
using caffe::Solver;
using caffe::SolverParameter;
using caffe::Caffe;
using caffe::caffe_scal;
using google::protobuf::io::CodedInputStream;
using std::string;
using std::vector;

// caffe cmd flags

DEFINE_int32(gpu, -1,
    "Run in GPU mode on given device ID.");
DEFINE_string(solver, "",
    "The solver definition protocol buffer text file.");
DEFINE_string(model, "",
    "The model definition protocol buffer text file..");
DEFINE_string(snapshot, "",
    "Optional; the snapshot solver state to resume training.");
DEFINE_string(workers, "W0",
    "cwd if workers, subdirectory of current directory.");
DEFINE_bool(fb_only, true,
    "DEPRECATED; workers only ForwardBackward.");
DEFINE_bool(synced, false,
    "DEPRECATED; pull/push synced with Forward");
// client puller / pusher flags
DEFINE_int32(pushstep, 3,
    "interval, in minibatches, between push operation.");
DEFINE_int32(pullstep, 3,
    "DEPRECATED interval, in minibatches, between pull operation.");

static std::shared_ptr<CaffeConfig> config;

void initCaffeConfig() {
  config.reset(new CaffeConfig(FLAGS_gpu, FLAGS_solver, FLAGS_model, FLAGS_snapshot, FLAGS_workers, FLAGS_fb_only, FLAGS_synced, FLAGS_pushstep, FLAGS_pullstep));
}

void enableP2P(int numGPUs){
  for (int i=0; i<numGPUs; i++){
    cudaSetDevice(i);
    for (int j=0; j<numGPUs; j++){
      int access;
      cudaDeviceCanAccessPeer(&access,i,j);
      if (access){
        cudaDeviceEnablePeerAccess(j,0);
        cudaGetLastError();
      }
    }
  }
}
void disableP2P(int numGPUs){
  for (int i=0; i<numGPUs; i++)
  {
    cudaSetDevice(i);

    for (int j=0; j<numGPUs; j++)
    {
      int access;
      cudaDeviceCanAccessPeer(&access, i, j);

      if (access)
      {
        cudaDeviceDisablePeerAccess(j);
        cudaGetLastError();
      }
    }
  }
}

std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}


std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, elems);
    return elems;
}

class CaffeWorker:public NetSolver{
private:

  std::mutex mu_update;
  std::condition_variable cv_update;
  bool start_update;

  std::mutex mu_weight; // protect write/read to weights

  std::mutex mu_diff;  //protect write to diffs diffCount
  int diffCount; // accumulated diff count

  caffe::Solver<float>* solver;

  volatile bool _terminate = false;

  std::vector<NetForwarder*> forwarders;

public:
  CaffeWorker(const string& name, const string& conf){
  }
  ~CaffeWorker(){
  }

  void init(){
    LL << "worker init()";
    start_update = false;
    solver = initCaffeSolver(-1, *config);
    //init shared parameter at worker
    for (int i = 0; i < solver->net()->params().size();i++){
      auto blob = solver->net()->params()[i];
    }

    //init forwarders
    vector<string> workerRoots = split(FLAGS_workers, ',');
    char* cwd = getcwd(nullptr,1024);
    LL << "cwd: " << cwd;
    CHECK(cwd != nullptr);
    string cwdString(cwd);
    for (int id = 0; id < workerRoots.size(); id++){
      bool display = id == 0;
      string workerRoot = cwdString + "/" + workerRoots[id];
      LL << "creating forwarder in: " << workerRoot;
//      CHECK(0 == chdir(workerRoot.c_str()));
      NetForwarder* forwarder = new NetForwarder(this, id, workerRoot, display, config.get());
      forwarders.push_back(forwarder);
      forwarder->startAsync();
    }
    enableP2P(forwarders.size());
//    CHECK(0 == chdir(cwd));
    free(cwd);
    LL << "worker init() over";
  }

  /**
   * by forwarder, accumulateDiff()
   */
  void signalUpdate(){
    std::unique_lock<std::mutex> l(mu_update);
    start_update = true;
    cv_update.notify_all();
  }

  /**
   * by worker
   */
  void waitUpdateSignal(){
    std::unique_lock<std::mutex> l(mu_update);
    while(!start_update){
      cv_update.wait(l);
    }
  }

  /**
   * by worker
   */
  void signalUpdateEnd(){
    std::unique_lock<std::mutex> l(mu_update);
    start_update = false;
    cv_update.notify_all();
  }


  /**
   * by forwarder, tryCopyWeight
   */
  void waitUpdateEndWeight(int wantedVersion) {
    std::unique_lock<std::mutex> l(mu_update);
    while(start_update || solver->iter() < wantedVersion) {
      cv_update.wait(l);
    }
  }

  void clearDiff(){
    for(int i = 0; i < solver->net()->params().size(); i++){
      auto blob = solver->net()->params()[i];
      switch(Caffe::mode()){
        case Caffe::CPU:
          caffe_set(blob->count(), (float)0, blob->mutable_cpu_diff());
          break;
        case Caffe::GPU:
          caffe_gpu_set(blob->count(), (float)0, blob->mutable_gpu_diff());
          break;
      }
    }
  }

  /**
   * by main
   */
  void run(){
    LL << "worker run()";
    LL << "initial pull over";
    for (int i = 0; i < forwarders.size(); i++){
      NetForwarder* forwarder = forwarders[i];
      forwarder->signalForward();
    }
    int iter = solver->param().max_iter() - solver->iter();
    for (int i = 0; i < iter; i++) {
      waitUpdateSignal();
      {
        Lock l(mu_diff);
        solver->ComputeUpdateValue();
        solver->net()->Update();
        clearDiff();
        solver->stepEnd();
      }
      signalUpdateEnd();
    }
    for (int i = 0; i < forwarders.size(); i++){
      NetForwarder* forwarder = forwarders[i];
      forwarder->joinForwardEnd();
    }
    disableP2P(forwarders.size());
    LL << "worker run() over";
  }

  /**
   * by forwarder
   */
  void gatherDiff(Solver<float>* another) {
    struct timeval tv;
    unsigned long long t0,t1,t2, t3, t4, t5;
    t0 = tick(&tv);
    Lock l(mu_diff);
    t1 = tick(&tv);
    for(int i = 0; i < another->net()->params().size(); i++){
      auto acc = solver->net()->params()[i];
      auto blob = another->net()->params()[i];
      ostringstream name;
      name << "gatherDiff:solver.blobs[" << i << "]";
//      checkNAN(blob->count(), blob->cpu_diff(), name.str());
      switch (Caffe::mode()) {
        case Caffe::CPU:
          caffe::caffe_add(acc->count(), blob->cpu_diff(), acc->cpu_diff(), acc->mutable_cpu_diff());
          break;
        case Caffe::GPU:
          caffe::caffe_gpu_add(acc->count(), blob->gpu_diff(), acc->gpu_diff(), acc->mutable_gpu_diff());
          break;
        default:
          LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
      }
//      caffe::caffe_add(acc->count(), blob->cpu_diff(), acc->cpu_diff(), acc->mutable_cpu_diff());
    }
    diffCount++;
    if(diffCount >= forwarders.size()) {
      signalUpdate();
    }
    t2 = tick(&tv);
    if(t2 - t0 > 100000){
      LL << "long accumulate diff:\tlock\t" << (t1-t0) << "\tadd\t" << (t2-t1);
    }
  }


  /**
   * by main
   */
  void pullIterations(Solver<float>* another) {
    another->setIter(solver->iter());
  }


  /**
   * by forwarder
   */
  void copyWeight(Solver<float>* another, int* version){
    Lock l(mu_weight); // lock weight, prevent pulling while copying
    float first,last;
    switch(Caffe::mode()){
      case Caffe::CPU:
        for (int i = 0; i < another->net()->params().size();i++){
          auto blob = another->net()->params()[i];
          float* dest = blob->mutable_cpu_data();
          auto src = solver->net()->params()[i];
          memcpy(dest, src->cpu_data(), blob->data()->size());
          if(i == 0){
            first = blob->cpu_data()[0];
          }else if(i == another->net()->params().size()-1){
            last = blob->cpu_data()[blob->count()-1];
          }
        }
        break;
      case Caffe::GPU:
        for (int i = 0; i < another->net()->params().size();i++){
          auto blob = another->net()->params()[i];
          float* dest = blob->mutable_gpu_data();
          auto src = solver->net()->params()[i];
          caffe_gpu_memcpy(blob->data()->size(), src->gpu_data(), dest);
        }
        break;
    }
    *version = solver->iter();
  }

  /**
   * by forwarder, check weight version & current wanted newest version number against worker's weight;
   * copy if newer version arrived;
   * mark
   */
  bool tryCopyWeight(Solver<float>* another, int* anotherCurrentVersion, int anotherWantedVersion){
    // synchronize with worker
    waitUpdateEndWeight(anotherWantedVersion);
    // need to copy
    copyWeight(another, anotherCurrentVersion);
    return true;
  }
};

int main(int argc, char *argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  initCaffeConfig();
  CaffeWorker worker("woker","");
  worker.init();
  worker.run();
  LL << "system exit";
  return 0;
}
