# DensitoAI 2.5.0 — файлы релиза

Эта ветка хранит только файлы поставки, код — в ветке `main`.

| Файл | Что это |
|---|---|
| `densitoai-2.5.0-image.tar.gz` | Docker-образ, 659 МБ (в распакованном виде 2,58 ГБ) |
| `densitoai-2.5.0-src.tar.gz` | исходники (git archive) |
| `SHA256SUMS` | контрольные суммы |
| `VERSION_MANIFEST.txt`, `*-WEIGHTS_SHA256.txt`, `*-image-id.txt`, `*-GIT_COMMIT.txt` | версия 2.5.0, config_hash cb4d9bc567e2, 25 файлов весов |

```
sha256sum -c SHA256SUMS
docker load < densitoai-2.5.0-image.tar.gz
docker run --rm --network none densitoai:2.5.0 verify
```

`*-GIT_COMMIT.txt` — коммит серверного репозитория команды; тот же код на GitHub — коммит `054c52c` ветки `main` (история очищена от DICOM образца организаторов, дерево файлов совпадает).
