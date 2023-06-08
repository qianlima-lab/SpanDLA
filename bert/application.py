"""
This examples clusters different sentences that come from the same wikipedia article.

It uses the 'wikipedia-sections' model, a model that was trained to differentiate if two sentences from the
same article come from the same section or from different sections in that article.
"""
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

embedder = SentenceTransformer('bert-base-wikipedia-sections-mean-tokens')

# Sentences and sections are from Wikipeda.
# Source: https://en.wikipedia.org/wiki/Bushnell,_Illinois
corpus = [
    (
        'The purpose of this study was to determine the acceptability of peer - and health-professional-led self-management education using the Stanford Program with Australian veterans and their partners .',
        'objective'),
    (
        'The @-week program taught problem-solving and decision-making skills to activate healthful behaviors , including action-planning and goal-setting .',
        'method'),
    (
        'The evaluation included a participant and facilitator postprogram questionnaire ; group interview ; and alcohol , posttraumatic stress disorder , anxiety , depression , anger , relationship , and quality-of-life measures as part of a randomized controlled study .',
        'method'),
    (
        'Participants included @ male veterans with comorbid alcohol dependency , psychiatric and medical conditions , and @ female partners ( n = @ ) , @ % of who reported a chronic condition .''result'),
    (
        'The primary outcome was a self-reported improvement in self-management of their conditions in @ % of participants , with another @ % reporting that their confidence to self-manage had improved .',
        'result'),
    ('There was an improvement in all measures at @ months .', 'result'),
    (
        'The program resulted in improvements in lifestyle and confidence in self-management for Vietnam veterans , a cohort difficult to engage in healthy behaviors .',
        'conclusion'),
    ('Most participants were also accompanied by their partners .', 'conclusion'),
    (
        'The program is a valuable resource for providing self-management education to veterans with alcohol dependency and various chronic conditions and needs to be considered in the suite of rehabilitation programs available to Defense Force personnel , veterans , and their partners .',
        'conclusion')
]

sentences = [row[0] for row in corpus]

corpus_embeddings = embedder.encode(sentences)
num_clusters = len(set([row[1] for row in corpus]))

# Sklearn clustering
km = AgglomerativeClustering(n_clusters=num_clusters)
km.fit(corpus_embeddings)

cluster_assignment = km.labels_

clustered_sentences = [[] for i in range(num_clusters)]
for sentence_id, cluster_id in enumerate(cluster_assignment):
    clustered_sentences[cluster_id].append(corpus[sentence_id])

for i, cluster in enumerate(clustered_sentences):
    print("Cluster ", i + 1)
    for row in cluster:
        print("(Gold label: {}) - {}".format(row[1], row[0]))
    print("")
