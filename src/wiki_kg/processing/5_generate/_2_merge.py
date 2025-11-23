# data ls, ordered by id, then take every 100 kg's, merge with kg.aggregate, then with deduplicate(semhash)
# do this in parallel, in multiproc, every 10000 articles, save the temporal results locally, store temps in gcp as well {from_id}_{to_id}.json
# then merge those aggregate, and 1 single semhash, store this in GCP
# .. could also generate embeddings of all the titles, and group every 10000 using knn, and generate graphs for topics, then merge them