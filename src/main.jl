# load data
using Distributions
W = [0, 1, 2, 3, 4]
X = [0 0 1 2 2;
     0 0 1 1 1;
     0 1 2 2 2;
     4 4 4 4 4;
     3 3 4 4 4;
     3 4 4 4 4]
W = W .+ 1
X = X .+ 1
N_D = size(X, 1)
N_W = length(W)
N_K = 2


α = 1
γ = 1
# topic assignment
# number of document x number of word per doc
Z = zeros(Int, N_D, N_W)
doc_topic_cnt = zeros(Int, N_D, N_K)
topic_term_cnt = zeros(Int, N_K, N_W)
for d = 1 : N_D
    for w = 1 : N_W
        # sample a topic
        z = rand(1 : N_K)
        Z[d, w] = z
        doc_topic_cnt[d, z] += 1
        topic_term_cnt[z, X[d, w]] += 1
    end
end

max_iter = 1000
for iter = 1 : max_iter
# sampling the topic assignment Z
# for every document, and for every word in that document
for d = 1 : N_D
    for w = 1 : N_W
        # sampling the topic for the word
        z = Z[d, w]
        doc_topic_cnt[d, z] -= 1
        topic_term_cnt[z, X[d, w]] -= 1
        # sample the topic
        new_p = (α + doc_topic_cnt[d, :]) .* (γ + topic_term_cnt[:, X[d, w]])
        new_p = new_p ./ sum(new_p)
        new_z_dist = Multinomial(1, new_p)
        new_z = find(rand(new_z_dist))[1]
        Z[d, w] = new_z
        doc_topic_cnt[d, new_z] += 1
        topic_term_cnt[new_z, X[d, w]] += 1
    end
end
end
# read out the document topic distribution
doc_topic_dist = zeros(N_D, N_K)

for d = 1 : N_D
    doc_topic_dist[d, :] = rand(Dirichlet(doc_topic_cnt[d, :] .+ α), 1)
end
# read out the topic term distribution
topic_term_dist = zeros(N_K, N_W)
for k = 1 : N_K
    topic_term_dist[k, :] = rand(Dirichlet(topic_term_cnt[k, :] .+ γ), 1)
end