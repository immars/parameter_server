/*
 * forwarder_solver.h
 *
 *  Created on: May 6, 2015
 *      Author: immars
 */

#ifndef SRC_APP_CAFFE_FORWARDER_SOLVER_H_
#define SRC_APP_CAFFE_FORWARDER_SOLVER_H_
#include <iostream>
#include <cuda.h>
#include "caffe/caffe.hpp"
#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/upgrade_proto.hpp"
#include "app/caffe/util.h"
#include "util/common.h"
using namespace caffe;
using namespace std;


class NetSolver {
public:
  virtual void copyWeight(Solver<float>* another, int* version) = 0;
  virtual bool tryCopyWeight(Solver<float>* another, int* anotherCurrentVersion, int anotherWantedVersion) = 0;
  virtual bool amendWeight(Solver<float>* another, float* estimatedVersion, unsigned long long forwardTime) { return false; };
  virtual void gatherDiff(Solver<float>* another, float version) = 0;
  virtual void pullIterations(Solver<float>* another) = 0;
  virtual ~NetSolver() {};
};


class NetForwarder {

  bool terminated;
  int id;
  NetSolver* worker;
  string rootDir;
  std::shared_ptr<CaffeConfig> config;
  caffe::Solver<float>* solver;
  int weightVersion; // current version
  int wantedVersion; // wanted version; increase with iterations
  float estimatedVersion; // estimated weight version on which forward/backward performed
  std::mutex mu_forward;
  std::condition_variable cv_forward;
  bool start_forward;
  std::unique_ptr<std::thread> internalThread;
  bool needDisplay;

  // for momentum prediction
  std::unique_ptr<Sequence<unsigned long long>> forwardTime;

public:
  NetForwarder(NetSolver* parent, int id, string workerRoot, bool display, CaffeConfig* config):
    id(id),worker(parent),rootDir(workerRoot),
    solver(nullptr),weightVersion(-1),wantedVersion(0),estimatedVersion(0),
    start_forward(false),needDisplay(display),terminated(false){
    this->config.reset(config);
    forwardTime.reset(new Sequence<unsigned long long>(8));
  }

  /**
   * by CaffeForwarder
   */
  void waitForwardSignal(){
    std::unique_lock<std::mutex> l(mu_forward);
    while(!start_forward){
      cv_forward.wait(l);
    }
  }

  /**
   * by CaffeForwarder
   */
  void signalForwardEnd(){
    std::unique_lock<std::mutex> l(mu_forward);
    start_forward = false;
    cv_forward.notify_all();
  }

  /**
   * by CaffeWorker
   */
  void signalForward() {
    std::unique_lock<std::mutex> l(mu_forward);
    start_forward = true;
    cv_forward.notify_all();
  }

  /**
   * by CaffeWorker
   */
  void joinForwardEnd() {
    if(!start_forward){
      return;
    }
    {
      std::unique_lock<std::mutex> l(mu_forward);
      while(start_forward) {
        cv_forward.wait(l);
      }
    }
  }

  void tryCopyWeight(){
    if(this->worker->tryCopyWeight(this->solver,
                                   &this->weightVersion,
                                   this->wantedVersion)){
      // copy successful; reset version counter to this newly copied version
      this->wantedVersion = this->weightVersion;
      this->estimatedVersion = this->weightVersion;
    }
    this->wantedVersion ++;
    if(!config->fb_only){
      this->worker->amendWeight(this->solver, &this->estimatedVersion, this->forwardTime->average());
    }
  }

  void accumulateDiff(){
    this->worker->gatherDiff(this->solver, this->estimatedVersion);
  }

  void pullIterations(){
    this->worker->pullIterations(this->solver);
  }

  void start() {
    struct timeval tv;
    unsigned long long t0, t1, t2, t3, t4, t5, t6;
    if(nullptr == solver) {
      solver = initSGDSolverInDir(id, rootDir, *config.get());
      LL << "Inited solver On device id # " << id;
    }
    int iter = solver->param().max_iter() - solver->iter();
    LL << "start training loop # " << id;
    waitForwardSignal();
    LL << "start() forward signal received";
    tryCopyWeight();
    pullIterations();
    for (int i = 0; i < iter; i++) {
      t0 = tick(&tv);
      // wait signal to forward
      if(needDisplay){
        solver->testPhase();
      }
      t1 = tick(&tv);
      tryCopyWeight();
      t2 = tick(&tv);
//      LL<< "forwarder # " << id;
      solver->forwardBackwardPhase();
      t3 = tick(&tv);
      this->accumulateDiff();
      t4 = tick(&tv);
      if(needDisplay){
        solver->displayPhase();
      }
      t5 = tick(&tv);
      // bypass all of computeUpdateValue if fb_only: forward_backward_only
      /*
      if(!config->fb_only){
        solver->ComputeUpdateValue();
        solver->net()->Update();
      }*/
      t6 = tick(&tv);
      solver->stepEnd();
      forwardTime->push(t3-t2);
      /*
      LL << "# " << id << "\ttestPhase\t"<< (t1-t0)
              << "\ttryCopyWeight\t"<< (t2-t1)
              << "\tforwardBackward\t"<< (t3-t2)
              << "\taccumulateDiff\t"<< (t4-t3)
              << "\tdisplayPhase\t"<< (t5-t4);
      */
    }
    LL << "Forwarder sending forward end signal";
    signalForwardEnd();
  }

  void startAsync(){
    if(!internalThread.get()){
      internalThread.reset(new thread(&NetForwarder::start, this));
    }
  }

  void stop() {
    //TODO
  }
};





#endif /* SRC_APP_CAFFE_FORWARDER_SOLVER_H_ */
