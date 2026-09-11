find_package(GTest REQUIRED)
include(GoogleTest)

add_executable(test_constraints test_constraints.cc ../iiwa_ik/constraints.cc ../iiwa_ik/iiwa_analytic_ik.cc)
target_link_libraries(test_constraints PRIVATE drake::drake GTest::gtest_main)
target_include_directories(test_constraints PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
gtest_discover_tests(test_constraints)

add_executable(test_costs test_costs.cc ../iiwa_ik/costs.cc ../iiwa_ik/iiwa_analytic_ik.cc)
target_link_libraries(test_costs PRIVATE drake::drake GTest::gtest_main)
target_include_directories(test_costs PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
gtest_discover_tests(test_costs)

add_executable(test_iiwa_analytic_ik test_iiwa_analytic_ik.cc ../iiwa_ik/iiwa_analytic_ik.cc)
target_link_libraries(test_iiwa_analytic_ik PRIVATE drake::drake GTest::gtest_main)
target_include_directories(test_iiwa_analytic_ik PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
gtest_discover_tests(test_iiwa_analytic_ik)

add_executable(test_parameterization test_parameterization.cc ../iiwa_ik/iiwa_analytic_ik.cc)
target_link_libraries(test_parameterization PRIVATE drake::drake GTest::gtest_main)
target_include_directories(test_parameterization PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
gtest_discover_tests(test_parameterization)

add_executable(test_gradients test_gradients.cc ../iiwa_ik/iiwa_analytic_ik.cc)
target_link_libraries(test_gradients PRIVATE drake::drake GTest::gtest_main)
target_include_directories(test_gradients PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
gtest_discover_tests(test_gradients)

add_executable(test_ift test_ift.cc ../iiwa_ik/constraints.cc ../iiwa_ik/iiwa_analytic_ik.cc ../iiwa_ik/costs.cc)
target_link_libraries(test_ift PRIVATE drake::drake GTest::gtest_main)
target_include_directories(test_ift PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
gtest_discover_tests(test_ift)

add_executable(test_newton_hessian test_newton_hessian.cc ../iiwa_ik/iiwa_analytic_ik.cc)
target_link_libraries(test_newton_hessian PRIVATE GTest::gtest drake::drake Eigen3::Eigen)
target_include_directories(test_newton_hessian PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
# Needed so the Drake cross-check test can locate the URDF and package.xml.
target_compile_definitions(test_newton_hessian PRIVATE
    IIWA_IK_REPO_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../..")
gtest_discover_tests(test_newton_hessian)

add_executable(test_regularization test_regularization.cc ../iiwa_ik/iiwa_analytic_ik.cc)
target_link_libraries(test_regularization PRIVATE drake::drake GTest::gtest_main)
target_include_directories(test_regularization PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
gtest_discover_tests(test_regularization)
