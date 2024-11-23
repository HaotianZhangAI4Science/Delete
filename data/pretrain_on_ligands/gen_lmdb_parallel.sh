#!/bin/zsh

# Define batch size and total count
BATCH_SIZE=100000
TOTAL_COUNT=1874685

# Calculate the number of full batches needed
FULL_BATCHES=$((TOTAL_COUNT / BATCH_SIZE))

# Run full batches in parallel
for i in $(seq 0 $((FULL_BATCHES - 1))); do
  START=$((i * BATCH_SIZE))
  END=$((START + BATCH_SIZE))
  python gen_lmdb_parallel.py \
  --cache_dir ./../data/cache \
  --output_dir ./../data/lmdb_batches \
  --start_idx $START \
  --end_idx $END &
done

# Process the remaining data
REMAINING_START=$((FULL_BATCHES * BATCH_SIZE))
if [ $REMAINING_START -lt $TOTAL_COUNT ]; then
  python gen_lmdb_parallel.py \
  --cache_dir ./../data/cache \
  --output_dir ./../data/lmdb_batches \
  --start_idx $REMAINING_START \
  --end_idx $TOTAL_COUNT &
fi

# Wait for all background tasks to complete
wait

echo "All batches processed"

: << 'COMMENT'
:cache_dir: The directory containing the sdf files
:output_dir: The directory to write the LMDB files to
:start_idx: The index of the first file to process
:end_idx: The index of the last file to process
COMMENT