file(GLOB OP_TESTS CONFIGURE_DEPENDS test_op/test_cu_*.cpp)
add_executable(test_llm test_main.cpp utils.cu
  test_tensor/test_buffer.cpp test_tensor/test_tensor.cpp
  ${OP_TESTS} test_runtime.cpp)
target_link_libraries(test_llm PRIVATE llama GTest::gtest)
target_include_directories(test_llm PRIVATE ${PROJECT_SOURCE_DIR})
set_target_properties(test_llm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME cpu_unit COMMAND test_llm --gtest_filter=-*cu*:*stream*:*align*:test_tensor.to_cpu)
add_test(NAME all_unit COMMAND test_llm)
set_tests_properties(all_unit PROPERTIES LABELS cuda)
