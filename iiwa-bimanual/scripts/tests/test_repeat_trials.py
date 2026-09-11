import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import os
import sys

# Add repo root to path so we can import scripts.experiments
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir   = os.path.abspath(os.path.join(script_dir, "../.."))
if repo_dir not in sys.path:
    sys.path.append(repo_dir)

from scripts.experiments.run_full_comparison import run_pipeline

class TestRepeatTrials(unittest.TestCase):
    def test_aggregation_logic(self):
        """
        Verifies that run_pipeline correctly aggregates results from multiple trials.
        """
        # Mock dependencies to avoid running actual heavy planning
        mock_meshcat = MagicMock()
        mock_checker = MagicMock()
        mock_plant = MagicMock()
        mock_diagram = MagicMock()
        
        cfg = {
            "name": "test_config",
            "short_name": "test",
            "use_ift": True,
            "old_reach": False,
            "use_psi": False,
            "exit_on_failure": False,
        }
        
        # Mock the stages
        with patch('scripts.experiments.run_full_comparison.run_iris_and_gcs') as mock_iris_gcs:
            mock_iris_gcs.return_value = (
                MagicMock(), # iris_options
                [],          # regions
                MagicMock(), # gcs_traj
                MagicMock(is_success=lambda: True), # gcs_result
                {"iris_total_time": 10.0, "gcs_solve_time": 2.0} # timings_partial
            )
            
            with patch('scripts.experiments.run_full_comparison.run_rrt_shortcut_trajopt_toppra') as mock_rrt_chain:
                # Return different results for trial 0 and trial 1 to test stats
                mock_rrt_chain.side_effect = [
                    {
                        "rrt_plan_time": 1.0,
                        "rrt_shortcut_time": 0.5,
                        "trajopt_solve_time": 0.8,
                        "rrt_success": True,
                        "trajopt_solve_success": True,
                        "toppra": {"GCS": 0.9, "RRT": 1.1, "Trajopt": 0.7},
                        "toppra_durations": {"GCS": 5.0, "RRT": 10.0, "Trajopt": 2.0},
                    },
                    {
                        "rrt_plan_time": 3.0,
                        "rrt_shortcut_time": 1.5,
                        "trajopt_solve_time": 1.2,
                        "rrt_success": True,
                        "trajopt_solve_success": False,
                        "toppra": {"GCS": 1.1, "RRT": 1.3, "Trajopt": 0.9},
                        "toppra_durations": {"GCS": 5.0, "RRT": 12.0, "Trajopt": 4.0},
                    }
                ]
                
                with patch('scripts.experiments.run_full_comparison.build_checker_and_plant') as mock_build:
                    mock_build.return_value = (mock_checker, mock_plant, mock_diagram)
                    
                    # Prevent file writing during tests
                    with patch('os.makedirs'), patch('json.dump'), patch('builtins.open', unittest.mock.mock_open()):
                        all_results = []
                        run_pipeline(cfg, mock_meshcat, all_results, num_trials=2, meshcat_save="none")
                        
                        self.assertEqual(len(all_results), 1)
                        res = all_results[0]
                        
                        # Timing Stats (Mean of [1.0, 3.0] is 2.0, Std is 1.0)
                        self.assertAlmostEqual(res["rrt_plan_time_mean"], 2.0)
                        self.assertAlmostEqual(res["rrt_plan_time_std"], 1.0)
                        
                        # Shortcut (Mean of [0.5, 1.5] is 1.0, Std is 0.5)
                        self.assertAlmostEqual(res["rrt_shortcut_time_mean"], 1.0)
                        self.assertAlmostEqual(res["rrt_shortcut_time_std"], 0.5)

                        # TrajOpt (Mean of [0.8, 1.2] is 1.0, Std is 0.2)
                        self.assertAlmostEqual(res["trajopt_solve_time_mean"], 1.0)
                        self.assertAlmostEqual(res["trajopt_solve_time_std"], 0.2)
                        
                        # TOPPRA solve time and trajectory duration are distinct
                        # quantities and must not be conflated: `toppra` is how long
                        # TOPPRA took, `toppra_durations` is how long the retimed
                        # trajectory lasts. The aggregation used to write durations
                        # into `toppra`, which every table labels as a runtime.
                        # Solve time (mean of [0.7, 0.9] is 0.8, std 0.1)
                        self.assertAlmostEqual(res["toppra"]["Trajopt"], 0.8)
                        self.assertAlmostEqual(res["toppra_Trajopt_std"], 0.1)
                        # Duration (mean of [2.0, 4.0] is 3.0, std 1.0)
                        self.assertAlmostEqual(res["toppra_durations"]["Trajopt"], 3.0)
                        self.assertAlmostEqual(res["toppra_duration_Trajopt_std"], 1.0)
                        
                        # Success Rates
                        self.assertEqual(res["rrt_success_rate"], 1.0)
                        self.assertEqual(res["trajopt_success_rate"], 0.5)

    def test_single_trial_compatibility(self):
        """
        Verifies that num_trials=1 results match trial 0 exactly (backward compatibility).
        """
        mock_meshcat = MagicMock()
        cfg = {"name": "test", "short_name": "test", "use_ift": True, "old_reach": False, "use_psi": False}
        
        with patch('scripts.experiments.run_full_comparison.run_iris_and_gcs') as mock_iris_gcs:
            mock_iris_gcs.return_value = (MagicMock(), [], MagicMock(), MagicMock(is_success=lambda: True), {})
            with patch('scripts.experiments.run_full_comparison.run_rrt_shortcut_trajopt_toppra') as mock_rrt_chain:
                trial_0_data = {
                    "rrt_plan_time": 1.23,
                    "toppra": {"RRT": 0.78},
                    "toppra_durations": {"RRT": 4.56}
                }
                mock_rrt_chain.return_value = trial_0_data
                with patch('scripts.experiments.run_full_comparison.build_checker_and_plant') as mock_build:
                    mock_build.return_value = (MagicMock(), MagicMock(), MagicMock())
                    with patch('os.makedirs'), patch('json.dump'), patch('builtins.open', unittest.mock.mock_open()):
                        results = []
                        run_pipeline(cfg, mock_meshcat, results, num_trials=1)
                        res = results[0]
                        self.assertEqual(res["rrt_plan_time"], 1.23)
                        self.assertEqual(res["toppra"]["RRT"], 0.78)
                        self.assertEqual(res["toppra_durations"]["RRT"], 4.56)

if __name__ == '__main__':
    unittest.main()
