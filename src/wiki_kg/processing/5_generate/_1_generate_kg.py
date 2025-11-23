# take josancamon/finewiki, for each row, generate a KG using KGGen in parallel
# save each one in GCP (datadrove)
# if already exists, skip
# semaphore or asyncio in parallel
# if article > 100k chars, do with chunking
# reference the file id in the huggingface finewiki version of the dataset