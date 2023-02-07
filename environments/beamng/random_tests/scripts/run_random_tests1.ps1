for ($num = 10 ; $num -ge 1 ; $num--)
{
   python .\run_random_tests.py --total_runs 150 --traffic_count $num --ai_mode limit --port 64256 --workspace BeamNGWorkspace
}