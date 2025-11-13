# List of disambiguation ids

Run this query

```
SELECT ?item WHERE {
  VALUES ?type { wd:Q4167410 wd:Q15623926 }  # disambig, set index
  ?item wdt:P31 ?type .
}
```

on https://query.wikidata.org/ and download to JSON. (disambiguation_sia_ids.json)

# Reference words

These were obtained by compiling the name of sections with highest link density per language and having an LLM filter them