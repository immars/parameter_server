/*
 * util.h
 *
 *  Created on: Apr 21, 2015
 *      Author: immars
 */

#ifndef SRC_APP_CAFFE_UTIL_H_
#define SRC_APP_CAFFE_UTIL_H_
#include <iostream>
#include <google/protobuf/io/coded_stream.h>
#include <glog/logging.h>
#include "util/common.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "caffe/caffe.hpp"
#include "caffe/common.hpp"

using namespace std;
using namespace caffe;

typedef std::lock_guard<std::mutex> Lock;

class CaffeConfig {
public:
  int gpu;
  string solver;
  string model;
  string snapshot;
  string workers;
  bool fb_only;
  bool synced;
  int pushstep;
  int pullstep;

  CaffeConfig(int gpu, const string& solver,
              const string& model,
              const string& snapshot,
              const string& workers,
              bool fb_only,
              bool synced,
              int pushstep,
              int pullstep):
                gpu(gpu),solver(solver),model(model),snapshot(snapshot)
                ,workers(workers),
                fb_only(fb_only), synced(synced),
                pushstep(pushstep),pullstep(pullstep){}
};

template <typename D>
class Sequence {
private:
  D* values;
  int head;
  int capacity;
  int count;
//  std::mutex mu;

public:
  Sequence(int capacity):
    head(0),capacity(capacity),count(0){
    if (capacity <= 0) {
      capacity = 1;
    }
    values = new D[capacity];
  }
  ~Sequence() {
    if(NULL != values) {
      delete values;
    }
  }

  void push(D v){
    values[head] = v;
    head = (head + 1) % capacity;
    if(count < capacity){
      count++;
    }
  }

  int getCount() {
    return count;
  }

  D average() {
    if (count == 0) {
      return 0;
    }
    D sum = 0;
    for(int i = 0; i < count; i++) {
      sum += values[i];
    }
    return sum / count;
  }

  D linear(int step) {
    if(count == 0) {
      return 0;
    }
    return values[head] + step * (values[head] - values[(head+capacity-count+1) % capacity]) / count;
  }
};

caffe::SolverParameter solver_param;

caffe::Net<float>* initCaffeNet(CaffeConfig& config){
  CHECK_GT(config.solver.size(), 0) << "Need a solver definition to train.";

  caffe::ReadProtoFromTextFileOrDie(config.solver, &solver_param);

  caffe::NetParameter net_param;
  std::string net_path = solver_param.net();
  caffe::ReadNetParamsFromTextFileOrDie(net_path, &net_param);
  return new caffe::Net<float>(net_param);
}

Solver<float>* initCaffeSolver(int id, CaffeConfig& config, bool sgd_only = false, bool snapshot = true){

  Solver<float>* solver;

  CHECK_GT(config.solver.size(), 0) << "Need a solver definition to train.";

  caffe::ReadProtoFromTextFileOrDie(config.solver, &solver_param);

  if (id < 0) {
    id = config.gpu;
  }

  if (id < 0
      && solver_param.solver_mode() == caffe::SolverParameter_SolverMode_GPU) {
    id = solver_param.device_id();
  }

  // Set device id and mode
  if (id >= 0) {
    LOG(INFO) << "Use GPU with device ID " << id;
    Caffe::SetDevice(id);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  if(!sgd_only){
    solver = caffe::GetSolver<float>(solver_param);
  }else{
    solver = new caffe::SGDSolver<float>(solver_param);
  }
  if (snapshot && config.snapshot.size()) {
    LOG(INFO) << "Resuming from " << config.snapshot;
    solver->Restore(config.snapshot.c_str());
  }

  return solver;
}

static std::mutex mu_pwd;
Solver<float>* initSGDSolverInDir(int id, string root, CaffeConfig& config){
  Lock l(mu_pwd);
  char* cwd = getcwd(nullptr,1024);
  LL << "previous cwd: " << cwd << " root: " << root;
  CHECK(cwd != nullptr);
  CHECK(0 == chdir(root.c_str()));
  Solver<float>* solver = initCaffeSolver(id, config, true, false);
  CHECK(0 == chdir(cwd));
  free(cwd);
  return solver;
}

void checkNAN(int count, const float* data, string blobName){
  bool isNan = false;
  int nanIndex = -1;
  int nanCount = 0;
  for (int j = 0; j < count; j++){
    if(isnan(data[j])){
      isNan = true;
      nanIndex = j;
      nanCount++;
    }
  }
  if(isNan){
    LL << nanCount << "NANs in "<< blobName <<"[" << nanIndex << "]!";
  }
}

inline unsigned long long tick(struct timeval* tv) {
  gettimeofday(tv, NULL);
  return tv->tv_sec * 1000000 + tv->tv_usec;
}

#endif /* SRC_APP_CAFFE_UTIL_H_ */
