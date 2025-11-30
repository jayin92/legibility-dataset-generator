import asyncio
import string
import os

# Configuration
CONCURRENT_RATING_JOBS = 10
PYTHON_EXECUTABLE = ".venv/bin/python" # Adjust if needed

async def run_command(cmd, name):
    print(f"[{name}] Starting: {cmd}")
    process = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()

    if process.returncode != 0:
        print(f"[{name}] Error:\n{stderr.decode()}")
        return False
    else:
        print(f"[{name}] Completed.")
        # print(f"[{name}] Output:\n{stdout.decode()}")
        return True

async def process_letter(letter, rating_semaphore):
    # 1. Generate Pairs
    generate_cmd = f"{PYTHON_EXECUTABLE} generate_pairs.py --letter {letter}"
    success = await run_command(generate_cmd, f"Generate {letter}")
    if not success:
        print(f"[{letter}] Generation failed. Skipping rating.")
        return

    # 2. Rate Pairs (with concurrency limit)
    csv_file = f"pairs_{letter}.csv"
    if not os.path.exists(csv_file):
        print(f"[{letter}] CSV file not found: {csv_file}")
        return

    # Construct rate_pairs command
    # Using defaults or what user likely wants:  gemini-2.5-flash-lite batch mode, 5-level scores
    rate_cmd = (
        f"{PYTHON_EXECUTABLE} rate_pairs.py {csv_file} "
        f"--provider gemini "
        f"--model gemini-2.5-flash-lite "
        f"--batch "
        f"--five-level-scores"
    )

    async with rating_semaphore:
        print(f"[{letter}] Acquiring slot for rating...")
        success = await run_command(rate_cmd, f"Rate {letter}")
        if not success:
             print(f"[{letter}] Rating failed.")

async def main():
    letters = string.ascii_lowercase # 'a' through 'z'
    rating_semaphore = asyncio.Semaphore(CONCURRENT_RATING_JOBS)
    
    tasks = []
    for letter in letters:
        tasks.append(process_letter(letter, rating_semaphore))
    
    print(f"Starting pipeline for {len(letters)} letters...")
    await asyncio.gather(*tasks)
    print("Pipeline complete.")

if __name__ == "__main__":
    asyncio.run(main())
