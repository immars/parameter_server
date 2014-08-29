#pragma once

#include <cstdlib>
#include <cstdio>
#include <string>
#include "zlib.h"
#include "util/integral_types.h"
#include "glog/logging.h"
#include "proto/config.pb.h"

#include "google/protobuf/descriptor.h"
#include "google/protobuf/message.h"
#include "google/protobuf/text_format.h"
#include "google/protobuf/io/tokenizer.h"

namespace PS {

class File {
 public:
  // Opens file "name" with flags specified by "flag".
  // Flags are defined by fopen(), that is "r", "r+", "w", "w+". "a", and "a+".
  static File* open(const std::string& name, const char* const flag);
  // If open failed, program will exit.
  static File* openOrDie(const std::string& name, const char* const flag);

  static File* openOrDie(const DataConfig& name, const char* const flag);

  static size_t size(const std::string& name);

  // Reads "size" bytes to buff from file, buff should be pre-allocated.
  size_t read(void* const buff, size_t size);

  // Reads "size" bytes to buff from file, buff should be pre-allocated.
  // If read failed, program will exit.
  // void ReadOrDie(void* const buff, size_t size);

  // Reads a line from file to a std::string.
  // Each line must be no more than max_length bytes
  char* readLine(char* const output, uint64 max_length);

  // Reads the whole file to a std::string, with a maximum length of 'max_length'.
  // Returns the number of bytes read.
  int64 readToString(std::string* const line, uint64 max_length);

  // Writes "size" bytes of buff to file, buff should be pre-allocated.
  size_t write(const void* const buff, size_t size);

  // Writes "size" bytes of buff to file, buff should be pre-allocated.
  // If write failed, program will exit.
  // void WriteOrDie(const void* const buff, size_t size);

  // Writes a std::string to file.
  size_t writeString(const std::string& line);

  // // Writes a std::string to file and append a "\n".
  // bool WriteLine(const std::string& line);

  // Closes the file.
  bool close();

  // Flushes buffer.
  // bool Flush();

  // Returns file size.
  size_t size();

  // // current file position
  // size_t position() {
  //   return (size_t) ftell(f_);
  // }

  // seek a position, starting from the head
  bool seek(size_t position);

  // Returns the file name.
  std::string filename() const { return name_; }

  // Deletes a file.
  static bool remove(const char* const name) { return remove(name) == 0; }

  // Tests if a file exists.
  static bool exists(const char* const name) { return access(name, F_OK) == 0; }

  // check if it is open
  bool open() const { return (is_gz_ ? gz_f_ != NULL : f_ != NULL); }

 private:
  File(FILE* f_des, const std::string& name)
      : f_(f_des), name_(name) { }
  File(gzFile gz_des, const std::string& name)
      : gz_f_(gz_des), name_(name) {
    is_gz_ = true;
  }

  FILE* f_ = NULL;
  gzFile gz_f_ = NULL;

  const std::string name_;
  bool is_gz_ = false;
};


// bool readFileToString(const std::string& file_name, std::string* output);
// bool writeStringToFile(const std::string& data, const std::string& file_name);

//// convenient functions dealing with protobuf

typedef google::protobuf::Message GProto;
bool readFileToProto(const DataConfig& file, GProto* proto);
void readFileToProtoOrDie(const DataConfig& file, GProto* proto);

bool readFileToProto(const std::string& file_name, GProto* proto);
void readFileToProtoOrDie(const std::string& file_name, GProto* proto);

bool writeProtoToASCIIFile(const GProto& proto, const std::string& file_name);
void writeProtoToASCIIFileOrDie(const GProto& proto, const std::string& file_name);

bool writeProtoToFile(const GProto& proto, const std::string& file_name);
void writeProtoToFileOrDie(const GProto& proto, const std::string& file_name);

// return the hadoop fs command
std::string hadoopFS(const HDFSConfig& conf);

// return the file names in a directory
std::vector<std::string> readFilenamesInDirectory(const std::string& directory);
std::vector<std::string> readFilenamesInDirectory(const DataConfig& directory);

// // locate the i-th file in *config*
// DataConfig ithFile(const DataConfig& config);

// return files matches the regex in *config*
DataConfig searchFiles(const DataConfig& config);
} // namespace PS
