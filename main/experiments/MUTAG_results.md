# MUTAG Model Comparison

| Model | Configuration | CV Accuracy |
|-------|---------------|---------------|
| GATV2 | batch=16, hidden=16, epochs=50, lr=0.001, 5-fold, processed RDF | 77.94% ± 6.17% |
| GINE | batch=32, hidden=64, epochs=100, lr=0.001, 10-fold, processed RDF | 76.18% ± 8.37% |
| GINE | batch=32, hidden=64, epochs=100, lr=0.001, 10-fold, normalization architecture, processed RDF | 78.82% ± 8.30% |
| PNA | batch=16, hidden=16, epochs=50, lr=0.001, 5-fold, collapsed dataset | 78.82% ± 5.39% |
| RGCN | batch=16, hidden=16, epochs=100, lr=0.001, 10-fold, collapsed dataset | 77.35% ± 7.68% |
| RGCN ON RDF VERSION | hidden=16, epochs=70, lr=0.005, 10-fold, filtered RDF | 76.77% ± 8.37% |
