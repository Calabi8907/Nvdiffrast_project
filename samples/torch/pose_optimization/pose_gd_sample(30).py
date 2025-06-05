import subprocess
import time
import argparse

def run_pose_optimization(object_name):
    command = [
        "python", 
        "samples/torch/pose_optimization/pose_gd.py",
        "--display-interval", "10",
        "--mip",
        "--max-iter", "500",
        "--object_name", object_name
    ]
    
    process = subprocess.Popen(command)
    return process.wait()

def main():
    parser = argparse.ArgumentParser(description='Run pose optimization 30 times')
    parser.add_argument("--object_name", type=str, required=True, help="Object name to optimize (e.g., book)")
    args = parser.parse_args()

    print("Starting 30 optimization runs...")
    
    for i in range(30):
        print(f"\n=== Starting run {i+1}/30 ===")
        start_time = time.time()
        
        exit_code = run_pose_optimization(args.object_name)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\nRun {i+1} completed in {duration:.2f} seconds")
        
        if i < 29:
            print("Waiting 5 seconds before next run...")
            time.sleep(5)
    
    print("\nAll 30 runs completed!")

if __name__ == "__main__":
    main()
