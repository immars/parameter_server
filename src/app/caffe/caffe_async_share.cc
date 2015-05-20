#include <iostream>
#include <unistd.h>
#include <chrono>
#include <thread>
#include <google/protobuf/io/coded_stream.h>
#include <sys/time.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "ps.h"
#include "system/app.h"
#include "parameter/v_vector.h"
#include "parameter/kv_vector.h"
#include "caffe/caffe.hpp"
#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/upgrade_proto.hpp"
#include "app/caffe/util.h"
#include "app/caffe/forwarder_solver.h"

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

#define V_WEIGHT "weight"
#define V_DIFF "diff"
#define V_SOLVER "solver"
#define V_ITER "iteration"
namespace PS {


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

class CaffeServer : public App, public VVListener<float>, public VVListener<char> {
 public:
  CaffeServer(const string& name, const string& conf) : App(name) {
    diffBlobFront = new std::vector<Blob<float>*>(); // for accumulate
    diffBlobBack = new std::vector<Blob<float>*>(); // for push
    start_update = false;
  }
  virtual ~CaffeServer() {
    for(auto blob : (*diffBlobFront)){
      delete blob;
    }
    for(auto blob : (*diffBlobBack)){
      delete blob;
    }
    for(auto blob : diffBlobs){
      delete blob;
    }
    delete diffBlobFront;
    delete diffBlobBack;
  }

  virtual void init() {
    LL << myNodeID() << ", this is server " << myRank();

    solver = initCaffeSolver(-1, *config);

    // initialize the weight at server
    int total_weight = 0;
    weights = new VVector<float>(V_WEIGHT, true, this);
    diffs = new VVector<float>(V_DIFF, false, this);
    solver_states = new VVector<char>(V_SOLVER, true, this);
    iterations = new VVector<int>(V_ITER, true, nullptr);
    iterations->value(0) = {solver->iter()};
    for (int i = 0; i < solver->net()->params().size();i++){
      auto blob = solver->net()->params()[i];
      weights->value(i).reset(blob->mutable_cpu_data(), blob->count(), false);
      auto newBlob = new Blob<float>(blob->num(), blob->channels(), blob->height(), blob->width());
      memset(newBlob->mutable_cpu_diff(), 0, newBlob->diff()->size());
      diffBlobFront->push_back(newBlob);
      newBlob = new Blob<float>(blob->num(), blob->channels(), blob->height(), blob->width());
      memset(newBlob->mutable_cpu_diff(), 0, newBlob->diff()->size());
      diffBlobBack->push_back(newBlob);

      newBlob = new Blob<float>(blob->num(), blob->channels(), blob->height(), blob->width());
      memset(newBlob->mutable_cpu_diff(), 0, newBlob->diff()->size());
      diffBlobs.push_back(newBlob);
      diffs->value(i).reset(diffBlobs[i]->mutable_cpu_diff(), diffBlobs[i]->count(), false);
      total_weight += blob->data()->size();
    }

    LL << "total weight size:" << total_weight;
  }
  
  void testPhase(){
    Lock l(mu_solver);
    solver->TestAll();
  }

  void snapshotPhase(){
    Lock l(mu_solver);
    solver->Snapshot();
  }

  void run() {
    LL << myNodeID() << ", server " << myRank() << " run()ing";
    while(true){
      if(diffCount == 0){
        waitUpdateSignal();
      }
      computeUpdate();
    }
    LL << myNodeID() << ", server " << myRank() << " over";
  }

  void process(const MessagePtr& msg) {
    auto sgd = msg->task.sgd();
    if (sgd.cmd() == SGDCall::UPDATE_MODEL) { // sync param to memory
    {
      Lock l(mu_solver);
    }
    }
  }

  /**
   * by run()
   */
  void waitUpdateSignal(){
    std::unique_lock<std::mutex> l(mu_update);
    while(!start_update){
      cv_update.wait(l);
    }
  }

  /**
   * by run()
   */
  void signalUpdateEnd(){
    std::unique_lock<std::mutex> l(mu_update);
    start_update = false;
    cv_update.notify_all();
  }

  /**
   * by vectorChanged()
   */
  void signalUpdate() {
    std::unique_lock<std::mutex> l(mu_update);
    start_update = true;
    cv_update.notify_all();
  }

  /**
   * by run()
   */
  void computeUpdate() {
    int count = 1;
    {
      Lock l(mu_diff);
      if(diffCount < 1){
        LL << "no diff accumulated!";
        return;
      }
      auto temp = diffBlobFront; diffBlobFront = diffBlobBack;diffBlobBack = temp;
      LL << "diff (" << diffCount << ") swapped to back";
      count = diffCount;
      diffCount = 0;
    }
    Lock l(mu_solver);
//    float first,last, firstv, lastv;
    for (int i = 0; i < solver->net()->params().size();i++){
      auto dest = solver->net()->params()[i];
      auto src = (*diffBlobBack)[i];
      // clear diffBlobBack
      memcpy(dest->mutable_cpu_diff(), src->cpu_diff(), dest->diff()->size());
      //scale down?
      if(count > 1){
        caffe::caffe_scal(dest->count(), float(1.0 / count), dest->mutable_cpu_diff());
      }
/*      if(i==0){
    first=blob->cpu_diff()[0];
    firstv = src[0];
      }else if(i == solver->net()->params().size()-1){
    last=blob->cpu_diff()[blob->count()-1];
    lastv = src[src.size() - 1];
      }
*/
    }

//    LL<< "got diff[" << first<<",...,"<<last<<"]/[" << firstv << ",...," << lastv <<"]";

    solver->ComputeUpdateValue();
    solver->net()->Update();
    solver->snapshotPhase();
    solver->stepEnd();

    for (auto blob : (*diffBlobBack)){
      memset(blob->mutable_cpu_diff(), 0, blob->diff()->size());
    }
    signalUpdateEnd();
  }

  void accumulateDiff() {
    Lock l(mu_diff);
    for(int i = 0; i < diffBlobFront->size();i++){
      auto src = diffs->value(i);
      auto dest = (*diffBlobFront)[i];
      caffe::caffe_add(dest->count(), src.data(), dest->cpu_diff(), dest->mutable_cpu_diff());
    }
    diffCount++;
    signalUpdate();
  }

  void vectorChanged(VVector<float>* data, MessagePtr& reply){
//    LL << "vector change received:" << data->name();
    CHECK_EQ(data, this->diffs) << "server only accept diff changes";
    int diffStartVersion = data->version();
    LL << "version: pushed\t" << diffStartVersion << "\tvs mine\t" << solver->iter();
    // CHECK_LE(diffStartVersion, solver->iter());
    int lag = solver->iter() - diffStartVersion;
    if(reply.get()){
      reply->task.set_version(solver->iter());
    }
    accumulateDiff();
    LL << "accumulate end at iter: " << solver->iter();
 }
  void vectorChanged(VVector<char>* data){
    CHECK(false) << "shouldn't be any VVector<char> change: "<< data->name();
  }

  void vectorGetting(VVector<float>* data){
    Lock l(mu_solver);
    float first, last;
    if (data == this->weights){
      // need to sync to CPU
      for (int i = 0; i < solver->net()->params().size();i++){
        auto blob = solver->net()->params()[i];
        blob->cpu_data();
        if (i==0) {
          first = blob->cpu_data()[0];
        } else if (i == solver->net()->params().size() - 1 ) {
          last = blob->cpu_data()[blob->count()-1];
        }
      }
      weights->setVersion(solver->iter());
      LL << "weight synced: ["<<first<<","<<last<<"]\t" << weights->version();
    } else {
      CHECK(false) << "some one is getting none-gettable! " << data->name();
    }
  }

  void vectorGetting(VVector<char>* data){
    LL << "getting char: "<< data->name();
    Lock l(mu_solver);
    if (data == this->solver_states){
      // need to serialize solver state into solver_states
      caffe::SolverState state;
      this->solver->SnapshotSolverState(&state);
      state.set_iter(solver->iter());
      state.set_current_step(solver->current_step());
      string buf;
      state.SerializeToString(&buf);
      solver_states->value(0).resize( buf.size() );
      memcpy(solver_states->value(0).data(), buf.data(), buf.size());
      LL << "server solver state saved, history:" << state.history_size() << ", total:" << buf.size();
    } else {
      CHECK(false) << "some one is getting none-gettable! " << data->name();
    }
  }


 private:
  VVector<char> *solver_states; // individual data ptr, solver state to initialize workers
  VVector<float> *weights; //share data ptr with solver->net->params->cpu_data

  VVector<float> *diffs; //individual data ptr with diffBlobs
  std::vector<Blob<float>*> diffBlobs; // for receiving from worker

  std::mutex mu_diff; // protect change to diffBlobFront and diffCount
  int diffCount; // how many diffBlobs accumulated in diffBlobFront
  std::vector<Blob<float>*>* diffBlobFront; // for accumulating from diffBlobs

  std::vector<Blob<float>*>* diffBlobBack; // for copying into solver

  std::mutex mu_solver;
  caffe::Solver<float>* solver;

  VVector<int> *iterations;

  std::mutex mu_update;
  std::condition_variable cv_update;
  bool start_update;

};

App* CreateServerNode(const std::string& conf) {
  return new CaffeServer("app", conf);
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

class CaffeWorker: public App, public NetSolver{
private:



  std::mutex mu_forward;
  std::condition_variable cv_forward;
  bool start_forward;

  std::mutex mu_push;
  std::condition_variable cv_push;

  std::mutex mu_pull;
  std::condition_variable cv_pull;
  bool start_pull;

  std::mutex mu_version; // protect change to weightVersion and requestedVersion
//  int weightVersion; // current version no. of weights, in iteration count
  int requestedVersion; // wanted version no. of weights, in iteration count

  std::mutex mu_weight; // protect write to weights
  VVector<float> *weights;// individual data ptr, same order/size as solver->net->params

  std::mutex mu_momentum; // protect write/read to momentum
  std::vector<Blob<float>*>* guessMomentum; // individual data ptr, guess momentum from pulled weights

  std::mutex mu_diff;  //protect write to diffs diffCount
  VVector<float> *diffs;// for accumulated diff, share memory with diffBuffer (front/end)
  int diffCount; // accumulated diff count
  float diffVersion;

  std::vector<Blob<float>*>* diffBlobFront; // for accumulate
  std::vector<Blob<float>*>* diffBlobBack; // for push
  caffe::Solver<float>* solver;

  VVector<int>* iterations;

  std::unique_ptr<std::thread> pusher;
  std::unique_ptr<std::thread> puller;

  volatile bool _terminate = false;

  std::vector<NetForwarder*> forwarders;

  std::unique_ptr<Sequence<float>> serverVersions;
  std::unique_ptr<Sequence<unsigned long long>> swapTimestamp;
  std::mutex mu_sequence;

public:
  CaffeWorker(const string& name, const string& conf):App(name){
//    weightVersion = 0;
    requestedVersion = 0;
    diffVersion = 0;
    diffBlobFront = new std::vector<Blob<float>*>();
    diffBlobBack = new std::vector<Blob<float>*>();
    guessMomentum = new std::vector<Blob<float>*>();

    serverVersions.reset(new Sequence<float>(8));
    swapTimestamp.reset(new Sequence<unsigned long long>(8));
  }
  ~CaffeWorker(){
    for(auto blob : (*diffBlobFront)){
      delete blob;
    }
    for(auto blob : (*diffBlobBack)){
      delete blob;
    }
    for(auto blob : (*guessMomentum)){
      delete blob;
    }

    delete diffBlobFront;
    delete diffBlobBack;
    delete guessMomentum;
  }

  void init(){
    LL << "worker init()";
    start_forward = false;
    start_pull = false;
    solver = initCaffeSolver(-1, *config);
    //init shared parameter at worker
    weights = new VVector<float>(V_WEIGHT);
    diffs = new VVector<float>(V_DIFF);
    iterations = new VVector<int>(V_ITER);
    iterations->value(0) = {0};
    for (int i = 0; i < solver->net()->params().size();i++){
      auto blob = solver->net()->params()[i];
      weights->value(i).resize(blob->count());
      auto newBlob = new Blob<float>(blob->num(), blob->channels(), blob->height(), blob->width());
      memset(newBlob->mutable_cpu_diff(), 0, newBlob->diff()->size());
      diffBlobFront->push_back(newBlob);
      newBlob = new Blob<float>(blob->num(), blob->channels(), blob->height(), blob->width());
      memset(newBlob->mutable_cpu_diff(), 0, newBlob->diff()->size());
      diffBlobBack->push_back(newBlob);
      diffs->value(i).reset((*diffBlobBack)[i]->mutable_cpu_diff(), (*diffBlobBack)[i]->count(), false);
      newBlob = new Blob<float>(blob->num(), blob->channels(), blob->height(), blob->width());
      memset(newBlob->mutable_cpu_diff(), 0, newBlob->diff()->size());
      memset(newBlob->mutable_cpu_data(), 0, newBlob->data()->size());
      guessMomentum->push_back(newBlob);
    }

    //init pusher/puller
    pusher.reset(new std::thread(&CaffeWorker::pusherMain, this));
    puller.reset(new std::thread(&CaffeWorker::pullerMain, this));

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

  void clearGuessMomentum(){
    std::unique_lock<std::mutex> l(mu_momentum);
    for(int i = 0; i < guessMomentum->size(); i++){
      auto blob = (*guessMomentum)[i];
      memset(blob->mutable_cpu_diff(), 0, blob->diff()->size());
    }
  }

  /**
   * by run() thread
   */
  void waitForwardSignal(){
    std::unique_lock<std::mutex> l(mu_forward);
    while(!start_forward){
      cv_forward.wait(l);
    }
  }

  /**
   * by run() thread
   */
  void signalForwardEnd(){
    std::unique_lock<std::mutex> l(mu_forward);
    start_forward = false;
    cv_forward.notify_all();
  }

  /**
   * by process() thread
   */
  void signalAndJoinForward() {
    std::unique_lock<std::mutex> l(mu_forward);
    start_forward = true;
    cv_forward.notify_all();
    while(start_forward) {
      cv_forward.wait(l);
    }
  }

  void pullerMain(){
    LL << "puller start";
    while(true){
      waitPullSignal();
      pullWeight();
      signalPullEnd();
    }
    LL << "puller exit";
  }

  void pusherMain(){
    LL << "pusher start";
    CUDA_CHECK(cudaSetDevice(0));
    while(true){
      waitPushSignal();
      pushDiff();
    }
    LL << "pusher exit";
  }

  /**
   * by pusher thread
   */
  void waitPushSignal(){
    std::unique_lock<std::mutex> l(mu_push);
    cv_push.wait(l);
    LL << "push signal received: " << diffCount;
  }

  void signalPush(){
    std::unique_lock<std::mutex> l(mu_push);
    LL << "signal push on: " << diffCount;
    cv_push.notify_all();
  }

  /**
   * by puller thread
   */
  void waitPullSignal(){
    std::unique_lock<std::mutex> l(mu_pull);
    while(!start_pull){
      cv_pull.wait(l);
    }
    LL << "pull signal received: " << requestedVersion << " vs " << weights->version();
  }

  /**
   * by puller thread
   */
  void signalPullEnd(){
    std::unique_lock<std::mutex> l(mu_pull);
    start_pull = false;
    cv_pull.notify_all();
  }

  /**
   * by worker run(), wait for initial pull
   */
  void waitPullEnd(){
    std::unique_lock<std::mutex> l(mu_pull);
    while(start_pull){
      cv_pull.wait(l);
    }
  }

  /**
   * by worker run() and forwarder.copy -> worker.tryCopyWeight()
   */
  void signalPull(){
    std::unique_lock<std::mutex> l(mu_pull);
    LL << "signal pull on: " << requestedVersion << " vs " << weights->version();
    start_pull = true;
    cv_pull.notify_all();
  }



  /**
   * by main
   */
  void run(){
    LL << "worker run()";
    this->requestedVersion = 0; // mark initial pull version as 0: default forwarder version is -1
    signalPull();
    waitPullEnd();
    clearGuessMomentum(); // initial momentum = 0
    LL << "initial pull over";
    for (int i = 0; i < forwarders.size(); i++){
      NetForwarder* forwarder = forwarders[i];
      forwarder->signalForward();
    }
    for (int i = 0; i < forwarders.size(); i++){
      NetForwarder* forwarder = forwarders[i];
      forwarder->joinForwardEnd();
    }
    disableP2P(forwarders.size());
    LL << "worker run() over";
  }

  void process(const MessagePtr& msg) {
    LL << "message received";
    auto sgd = msg->task.sgd();
    if (sgd.cmd() == SGDCall::UPDATE_MODEL) { // sync param to memory
      LL << "process() update model received";
      signalAndJoinForward();
      LL << "process() forward end received";
    }
  }

  /**
   * by forwarder
   */
  void gatherDiff(Solver<float>* another, float version) {
    struct timeval tv;
    unsigned long long t0,t1,t2, t3, t4, t5;
    t0 = tick(&tv);
    Lock l(mu_diff);
    t1 = tick(&tv);
    for(int i = 0; i < another->net()->params().size(); i++){
      auto acc = (*diffBlobFront)[i];
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
    LL << "gatherDiff:old\t" << diffVersion <<"\tnew\t" << version;
    diffVersion = (diffVersion * diffCount + version) / (diffCount + 1);
    diffCount++;
    if(diffCount >= FLAGS_pushstep) {
      signalPush();
    }
    t2 = tick(&tv);
    if(t2 - t0 > 100000){
      LL << "long accumulate diff:\tlock\t" << (t1-t0) << "\tadd\t" << (t2-t1);
    }
  }

  /**
   * by pusher, synchronized (block until message sent)
   */
  void pushDiff(){
    struct timeval tv;
    unsigned long long swapTime = 0;
    {
      // copy diff to diffBuffer
      Lock l(mu_diff);
      float first, last;
      auto temp = diffBlobFront; diffBlobFront = diffBlobBack; diffBlobBack = temp; // for accumulate
      LL << "Worker diff("<< diffCount <<") swapped to back";
      // clear diff count
      diffCount = 0;
      swapTime = tick(&tv);
    }
    // reset diffs Vector pointer; sync blob diff to cpu
    for(int i = 0; i < diffBlobBack->size(); i++){
      auto blob = (*diffBlobBack)[i];
      diffs->value(i).reset(blob->mutable_cpu_diff(), blob->count(), false);
    }
    //push to app instead of
    MessagePtr msg(new Message(kServerGroup));
    msg->key = {0};
    msg->task.set_key_channel(0);
    /*
    for(int i = 0; i < diffs->vcount();i++){
      auto acc = (*diffBlobBack)[i];
      acc->cpu_diff(); // sync to cpu
      auto diff = diffs->value(i);
      CHECK_EQ(acc->cpu_diff(), diff.data());
    }*/
    diffs->setVersion(diffVersion);
    diffs->getValue(msg);
    int push_time = diffs->push(msg);
    LL << "begin diff waitOutMsg";
    diffs->waitOutMsg(kServerGroup, push_time);
    {
      Lock l_seq(mu_sequence);
      serverVersions->push(diffs->version());
      swapTimestamp->push(swapTime);
      LL << "diffVersion:\t" << diffVersion << "\tserverVersion:\t" << diffs->version();
    }
    //clear previous diff
    for(auto acc : (*diffBlobBack)){
      switch(Caffe::mode()){
        case Caffe::CPU:
          memset(acc->mutable_cpu_diff(), 0, acc->diff()->size());
          break;
        case Caffe::GPU:
          caffe_gpu_set(acc->count(), (float)0, acc->mutable_gpu_diff());
          break;
      }
    }

    LL << "Worker diff pushed to server";
  }

  /**
   * by puller (except the initial pull), synchronized
   */
  void pullWeight(){
    LL << "begin pull weight";

    Lock l(mu_weight);
    if(!config->fb_only){
      Lock l_mom(mu_momentum);
      // save last weight to calculate momentum
      for(int i = 0; i < guessMomentum->size(); i++){
        auto src = weights->value(i);
        auto destBlob = (*guessMomentum)[i];
        memcpy(destBlob->mutable_cpu_data(), src.data(), destBlob->data()->size());
      }
    }
    int lastWeightVersion = weights->version();
    MessagePtr msg(new Message(kServerGroup));
    msg->key = {0};
    LL << "begin pull";
    int pull_time = weights->pull(msg);
    LL << "begin waitOutMsg";
    weights->waitOutMsg(kServerGroup, pull_time);

    if(!config->fb_only){
      // calculate momentum
//      int serverUpdates = forwarders.size() * config->pullstep * this->sys_.yp().num_workers() / config->pushstep;
      int serverUpdates = weights->version() - lastWeightVersion;
      LL << "guessing momentum from weight delta\t" << serverUpdates << "\t" << weights->version();
      if (serverUpdates > 0) {
        Lock l_mom(mu_momentum);
        for(int i = 0; i < guessMomentum->size(); i++){
          auto blob = (*guessMomentum)[i];
          const float* last = blob->cpu_data();
          float* next = weights->value(i).data();
          float* momentum = blob->mutable_cpu_diff();
          caffe_sub(blob->count(), last, next, momentum);
          caffe_scal(blob->count(), (float)1.0 / serverUpdates, momentum);
        }
      }
    }

    LL << "weight pulled from server, total:" << weights->totalSize();
  }

  /**
   * by main
   */
  void pullIterations(Solver<float>* another) {
    MessagePtr msg(new Message(kServerGroup));
    msg->key = {0};
    int pull_time = iterations->pull(msg);
    iterations->waitOutMsg(kServerGroup, pull_time);
    Lock l(mu_weight);
    SArray<int>src = iterations->value(0);
    LL << "iteration got: " << src.size() << "," << src[0];
    another->setIter(src[0]);
  }


  /**
   * by forwarder
   */
  void copyWeight(Solver<float>* another, int* version){
    Lock l(mu_weight); // lock weight, prevent pulling while copying
    float first,last;
    for (int i = 0; i < another->net()->params().size();i++){
      auto blob = another->net()->params()[i];
      float* dest = blob->mutable_cpu_data();
      auto src = weights->value(i);
      memcpy(dest, src.data(), blob->data()->size());
      //TODO direct copy to GPU?
      if(i == 0){
        first = blob->cpu_data()[0];
      }else if(i == another->net()->params().size()-1){
        last = blob->cpu_data()[blob->count()-1];
      }
      /*
      if(!config->fb_only){
        SGDSolver<float>* sgd = (SGDSolver<float>*) another;
        auto momentum = sgd->history()[i];
        auto guessed = (*guessMomentum)[i];
        memcpy(momentum->mutable_cpu_data(), guessed->cpu_diff(), guessed->diff()->size());
      }
      */
    }
    *version = weights->version();
    LL << "weight from server:[" << first << ",...," << last << "]";
  }

  /**
   * by forwarder, check weight version & current wanted newest version number against worker's weight;
   * copy if newer version arrived;
   * mark
   */
  bool tryCopyWeight(Solver<float>* another, int* anotherCurrentVersion, int anotherWantedVersion){
    if(requestedVersion < anotherWantedVersion){
      // mark newer version requested
      Lock l(mu_version);
      if(requestedVersion < anotherWantedVersion){
        requestedVersion = anotherWantedVersion;
        if(requestedVersion - weights->version() >= FLAGS_pullstep){
          signalPull();
        }
      }
    }
    if(weights->version() <= *anotherCurrentVersion){
      // no need to copy
      return false;
    }
    // need to copy
    copyWeight(another, anotherCurrentVersion);
    return true;
  }

  unsigned long long nextPushTime(int step) {
    return swapTimestamp->linear(step);
  }

  float nextPushWeight(int step) {
    return this->serverVersions->linear(step);
  }

  bool amendWeight(Solver<float>* another, float* estimatedVersion, unsigned long long forwardTime) {
    struct timeval tv;
    unsigned long long now = tick(&tv);
    unsigned long long nextPush = nextPushTime(1);
    float nextPushVersion = nextPushWeight(1);
    if(nextPush == 0 || nextPushVersion == 0
        || serverVersions->getCount() <= 1 // cannot reliably predict
        || swapTimestamp->getCount() <= 1 // cannot reliably predict
        || (now + forwardTime < nextPush && nextPushVersion - *estimatedVersion < 0.5)) {
      // no need to amend
      return false;
    }
    {
      Lock l(mu_momentum); // lock weight, prevent pulling while copying
      Lock l_seq(mu_sequence);
      now = tick(&tv);
      unsigned long long nextFBEnd = now + forwardTime;
      int step = 1;
      for(; true ; step++){
        nextPush = nextPushTime(step);
        nextPushVersion = nextPushWeight(step);
        if(nextFBEnd < nextPush || step > config->pullstep){
          break;
        }
      }
      float deltaVersion = nextPushVersion - *estimatedVersion;
      LL << "weight amend: now\t" << now << "\tnextFBEnd\t" << nextFBEnd
          << "\tnextPush\t" << nextPush
          << "\tmyguess\t" << *estimatedVersion
          << "\tnextVersion\t" << nextPushVersion
          << "\tstep\t" << step
          << "\tdelta\t" << deltaVersion;
      if(deltaVersion > 0){
        for (int i = 0; i < another->net()->params().size();i++){
          auto blob = another->net()->params()[i];
          float* weight = blob->mutable_cpu_data();
          const float* momentum = (*guessMomentum)[i]->cpu_diff();
          caffe_axpy(blob->count(), -deltaVersion, momentum, weight);
        }
        *estimatedVersion = nextPushVersion;
      }
    }
    return true;
  }

};

} // namespace PS

namespace PS {
App* App::create(const string& name, const string& conf) {
  auto my_role = Postoffice::instance().myNode().role();
  if (my_role == Node::SERVER) {
    return new CaffeServer(name, conf);
  } else if(my_role == Node::WORKER){
      return new CaffeWorker(name, conf);
  }else{
    return new App();
  }
}
} // namespace PS


int main(int argc, char *argv[]) {

  google::ParseCommandLineFlags(&argc, &argv, true);

  initCaffeConfig();

  auto& sys = PS::Postoffice::instance();
  sys.start(&argc, &argv);

  sys.stop();
  LL << "system exit";
  return 0;
}

