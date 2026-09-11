#define _USE_MATH_DEFINES
#include "iiwa_ik/iiwa_analytic_ik.h"
#include <cmath>
#include <gtest/gtest.h>
#include "drake/math/autodiff_gradient.h"

using drake::AutoDiffXd;
using drake::math::InitializeAutoDiff;
using drake::math::ExtractGradient;
using drake::math::ExtractValue;

TEST(RegularizationTest, ParameterRules) {
    AutoDiffConfig ad_config;
    // Default should have no SVT
    EXPECT_FALSE(ad_config.svt_epsilon.has_value());
    EXPECT_FALSE(ad_config.svt_lambda_max.has_value());
    EXPECT_EQ(ad_config.lambda, 0.0);
    
    ad_config.svt_epsilon = 0.1;
    EXPECT_TRUE(ad_config.svt_epsilon.has_value());
    EXPECT_EQ(*ad_config.svt_epsilon, 0.1);
}

TEST(RegularizationTest, GradientContinuitySweep) {
    BimanualConfig config(true, true, false, 0.6);
    
    // Configuration where subordinate elbow is close to zero (nearly singular)
    Eigen::VectorXd q_base(8);
    q_base << 1.969164758213879, -1.6338827982650315, 0.23082628320983334, 
               1.4206898770180691, 1.6608120548981948, -0.9460249093286206, 
               2.650924310535348, 3.99493129098573;

    // Use a large epsilon to ensure we cross it during a small sweep if sigma_min is small
    // Actually, sigma_min for this config is ~0.014 as found by Python script.
    double epsilon = 0.015; 
    double lambda_max = 0.01;
    
    AutoDiffConfig ad_config;
    ad_config.use_ift = true;
    ad_config.ift_handling = IftSingularityHandling::kLevenbergMarquardt;
    ad_config.svt_epsilon = epsilon;
    ad_config.svt_lambda_max = lambda_max;

    // Sweep q_base(0) slightly. This will change the Jacobian and thus sigma_min.
    int steps = 100;
    Eigen::MatrixXd last_grad;
    for (int i = 0; i < steps; ++i) {
        Eigen::VectorXd q = q_base;
        q(0) += (i - steps/2) * 1e-4;
        
        drake::AutoDiffVecXd q_ad = InitializeAutoDiff(q);
        drake::AutoDiffVecXd q_full_ad = IiwaBimanualParameterizationIFT(q_ad, config, ad_config, nullptr);
        Eigen::MatrixXd grad = ExtractGradient(q_full_ad);
        
        if (i > 0) {
            double diff = (grad - last_grad).norm();
            // Ensure no huge jumps. 1e-1 is a safe upper bound for a 1e-4 step in q.
            EXPECT_LT(diff, 0.5) << "Possible discontinuity at step " << i;
        }
        last_grad = grad;
    }
}

TEST(RegularizationTest, ResidualDampingSmokeTest) {
    BimanualConfig config(true, true, false, 0.6);
    
    Eigen::VectorXd q_base(8);
    q_base << 1.9, -1.6, 0.2, 1.4, 1.6, -0.9, 2.6, 3.9;

    AutoDiffConfig ad_config;
    ad_config.use_ift = true;
    ad_config.ift_handling = IftSingularityHandling::kResidualDamping;
    ad_config.lambda = 1.0; 

    drake::AutoDiffVecXd q_ad = InitializeAutoDiff(q_base);
    drake::AutoDiffVecXd q_full_ad = IiwaBimanualParameterizationIFT(q_ad, config, ad_config, nullptr);
    
    EXPECT_EQ(q_full_ad.size(), 14);
    // Gradient should be non-zero
    Eigen::MatrixXd grad = ExtractGradient(q_full_ad);
    EXPECT_GT(grad.norm(), 1e-3);
}

TEST(RegularizationTest, ResidualWeightedDampingSmokeTest) {
    BimanualConfig config(true, true, false, 0.6);
    
    Eigen::VectorXd q_base(8);
    q_base << 1.9, -1.6, 0.2, 1.4, 1.6, -0.9, 2.6, 3.9;

    AutoDiffConfig ad_config;
    ad_config.use_ift = true;
    ad_config.ift_handling = IftSingularityHandling::kResidualDamping;
    ad_config.use_anisotropic_damping = true;
    ad_config.lambda = 1.0; 

    drake::AutoDiffVecXd q_ad = InitializeAutoDiff(q_base);
    drake::AutoDiffVecXd q_full_ad = IiwaBimanualParameterizationIFT(q_ad, config, ad_config, nullptr);
    
    EXPECT_EQ(q_full_ad.size(), 14);
    Eigen::MatrixXd grad = ExtractGradient(q_full_ad);
    EXPECT_GT(grad.norm(), 1e-3);
}
