1. use gcsfs for file management instead of local
2. log all scripts logs into a .log file in this directory, every script run should append
3. handle retries, and failures from batch api in specific requests
4. handle chunking in relations
5. ensure it works at scale, (_5_get graphs nor _6_merge are ready to be functional at scale)