# Generate shape-based scoring results

`shape_score.py` is a script for scoring molecular docking poses. The main functions of this script include:

- Reading original and docked SDF files
- Calculating shape similarity scores
- Sorting docking results
- Saving processed results as PKL files
- Logging failed target paths

The script logs detailed processing steps and outputs all failed target paths at the end of processing.

## Environment Dependencies

You can install the required Python libraries using `pip`:

```bash
pip install rdkit easydict
```

## Usage

### Configuring Paths

Edit the following variables in the `shape_score.py` file to set your working directory and task name（here is just an example）:

```python
base_path = '/home/megagatlingpea/workdir/delete_shape'
task = 'nabla510_finetune_293'
```

Ensure that the corresponding task directory exists under `base_path`, and each task directory contains an `SDF` subdirectory and original SDF files.

### Running the Script

It's recommended to use the `nohup` command to run the script in the background, allowing it to continue executing even if the terminal is closed. Run the command as follows:

```bash
nohup python shape_score.py > process.log 2>&1 &
```

- `nohup`: Ignores the hangup signal, allowing the script to run continuously in the background.
- `python process.py`: Executes the script.
- `> process.log`: Redirects standard output to the `process.log` file.
- `2>&1`: Redirects standard error to standard output.
- `&`: Puts the command in the background.

### Viewing Logs

You can view the contents of the log file in real-time to monitor the script's execution:

```bash
tail -f process.log
```

### Checking Failed Target Paths

The script records all failed target paths during processing and outputs them in the last line of the log file. You can use the `eval` function to read the `failed` list from the last line of the `process.log` file to easily see which paths failed processing.

```python
with open('process.log', 'r') as f:
    lines = f.readlines()
    last_line = lines[-1]
    failed_targets = eval(last_line.split(':')[-1].strip())
    print("Failed target paths:")
for target in failed_targets:
    print(target)
```
