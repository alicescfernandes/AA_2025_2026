1. Aplicar K-Means como método base de clustering.
2. Aplicar um segundo método (AgglomerativeClustering) para comparação.
3. Avaliar e comparar os métodos considerando:
    - separação dos clusters (silhouette)
    - distância entre centróides
    - palavras representativas
    - consistência visual (PCA)
    - distribuição dos clusters
4. Investigar o impacto da variação de k no desempenho e interpretação dos clusters.
5. Escolher o método e valor de k mais adequado com base nas métricas.


#### 2.3.2.2 Dendrograma (Apagar)

```
# ~50m
Link=AgglomerativeClustering(n_clusters=None, linkage="complete",distance_threshold=0).fit(X_dense)
plot_dendrogram(Link, truncate_mode="level", p=5)
```