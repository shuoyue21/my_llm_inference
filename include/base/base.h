#ifndef FORWARD_INCLUDE_BASE_BASE_H_
#define FORWARD_INCLUDE_BASE_BASE_H_
#include <glog/logging.h>
#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string>
#define UNUSED(expr) \
  do {               \
    void(expr);      \
  } while (0)

namespace model {
enum class ModelBufferType {
  kInputTokens = 0,
  kInputEmbeddings = 1,
  kOutputRMSNorm = 2,
  kKeyCache = 3,
  kValueCache = 4,
  kQuery = 5,
  kInputPos = 6,
  kScoresStorge = 7,
  kOutputMHA = 8,
  kAttnOutput = 9,
  kW10output = 10,
  kW20output = 11,
  kW30output = 12,
  kFFRMSNorm = 13,
  kForwardOutput = 15,
  kForwardOutputCpu = 16,

  kSinCache = 17,
  kCosCache = 18,
};
}

namespace base {
enum class DeviceType : uint8_t {
  kDeviceUnknown = 0,
  kDeviceCPU = 1,
  kDeviceCUDA = 2,
};

enum class DataType : uint8_t {
  kDataTypeUnknown = 0,
  kDataTypeFp32 = 1,
  kDataTypeInt8 = 2,
  kDataTypeInt32 = 3,
};

enum class ModelType : uint8_t {
  kModelTypeUnknown = 0,
  kModelTypeLLama2 = 1,
};

inline size_t DataTypeSize(DataType data_type) {
  if (data_type == DataType::kDataTypeFp32) {
    return sizeof(float);
  } else if (data_type == DataType::kDataTypeInt8) {
    return sizeof(int8_t);
  } else if (data_type == DataType::kDataTypeInt32) {
    return sizeof(int32_t);
  } else {
    return 0;
  }
}

class NoCopyable {
 protected:
  NoCopyable() = default;

  ~NoCopyable() = default;

  NoCopyable(const NoCopyable&) = delete;

  NoCopyable& operator=(const NoCopyable&) = delete;
};

enum StatusCode : uint8_t {
  kSuccess = 0,
  kFunctionUnImplement = 1,
  kPathNotVaild = 2,
  kModeParseError = 3,
  kInternalError = 5,
  kKeyValueHasExits = 6,
  kInvaildArgument = 7,
};

enum class TokenizerType {
  kEncodeUnknown = -1,
  kEncodeSpe = 0,
  kEncodeBpe = 1,
};

class Status {
 public:
  Status(int code = StatusCode::kSuccess, std::string_view err_message = "");

  Status(const Status& other) = default;

  Status& operator=(const Status& other) = default;

  Status& operator=(int code);

  bool operator==(int code) const;

  bool operator!=(int code) const;

  operator int() const;

  operator bool() const;

  int32_t get_err_code() const;
  const std::string& get_err_msg() const;

  void set_err_msg(const std::string& err_msg);

 private:
  int code_ = StatusCode::kSuccess;
  std::string message_;
};

namespace error {
#define STATUS_CHECK(call)                                                                 \
  do {                                                                                     \
    const base::Status& status = call;                                                     \
    if (!status) {                                                                         \
      const size_t buf_size = 512;                                                         \
      char buf[buf_size];                                                                  \
      snprintf(buf, buf_size - 1,                                                          \
               "Infer error\n File:%s Line:%d\n Error code:%d\n Error msg:%s\n", __FILE__, \
               __LINE__, int(status), status.get_err_msg().c_str());                       \
      LOG(FATAL) << buf;                                                                   \
    }                                                                                      \
  } while (0)

Status Success(std::string_view err_msg = "");
Status FunctionNotImplement(std::string_view err_msg = "");
Status PathNotVaild(std::string_view err_msg = "");
Status ModeParseError(std::string_view err_msg = "");
Status InternalError(std::string_view err_msg = "");
Status KeyHasExits(std::string_view err_msg = "");
Status InvaildArgument(std::string_view err_msg = "");

}  // namespace error

std::ostream& operator<<(std::ostream& os, const Status& x);

}  // namespace base

#endif  // FORWARD_INCLUDE_BASE_BASE_H
