#include "base/base.h"
#include <string>

namespace base {
Status::Status(int code, std::string_view err_msg) : code_(code), message_(err_msg) {}

Status& Status::operator=(int code) {
  code_ = code;
  return *this;
}

bool Status::operator==(int code) const {
  if (code_ == code) {
    return true;
  } else {
    return false;
  }
}
bool Status::operator!=(int code) const {
  if (code_ != code) {
    return true;
  } else {
    return false;
  }
}
Status::operator int() const { return code_; }

Status::operator bool() const { return code_ == kSuccess; }

int32_t Status::get_err_code() const { return code_; }

const std::string& Status::get_err_msg() const { return message_; }

void Status::set_err_msg(const std::string& msg) { message_ = msg; }

namespace error {
Status Success(std::string_view err_msg) { return Status{kSuccess, err_msg}; }
Status FunctionNotImplement(std::string_view err_msg) {
  return Status{kFunctionUnImplement, err_msg};
}
Status PathNotVaild(std::string_view err_msg) { return Status{kPathNotVaild, err_msg}; }
Status ModeParseError(std::string_view err_msg) { return Status{kModeParseError, err_msg}; }
Status InternalError(std::string_view err_msg) { return Status{kInternalError, err_msg}; }
Status KeyHasExits(std::string_view err_msg) { return Status{kKeyValueHasExits, err_msg}; }
Status InvaildArgument(std::string_view err_msg) { return Status{kInvaildArgument, err_msg}; }
}  // namespace error

std::ostream& operator<<(std::ostream& os, const Status& x) {
  os << x.get_err_msg();
  return os;
}
}  // namespace base