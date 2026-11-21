# Google Colab T4 GPU Fine-Tuning Kılavuzu (Görev T4 / E3)

Bu kılavuz, **Whisper-Tiny** modelini Google Colab T4 GPU kullanarak Swahili FLEURS eğitim setiyle fine-tune etme ve sonuçları değerlendirme adımlarını içerir.

---

## 1. Colab Çalışma Zamanı Ayarı
1. [Google Colab](https://colab.research.google.com)'ı açın ve yeni bir not defteri oluşturun.
2. Üst menüden **Runtime (Çalışma Zamanı) > Change runtime type (Çalışma zamanı türünü değiştir)** seçeneğine tıklayın.
3. Donanım hızlandırıcı olarak **T4 GPU** seçin ve **Kaydet**'e basın.
4. GPU'nun aktif olduğunu doğrulamak için ilk hücrede çalıştırın:
   ```bash
   !nvidia-smi
   ```

---

## 2. Depoyu Klonlama ve Gereksinimleri Yükleme
```bash
# 1. Depoyu klonlayın ve klasöre geçin
!git clone https://github.com/BeyzaESER22/swahili-asr-finetuning.git
%cd swahili-asr-finetuning

# 2. Temizlenmiş ses dosyalarını Git LFS ile çekin (~2.1 GB)
!git lfs pull --include="data/cleaned/audio/*"

# 3. Model eğitimi ve çıkarım gereksinimlerini kurun
!pip install -r requirements-models.txt
```

---

## 3. Hızlı 2 Örnekli Deneme (Smoke Test)
Tam eğitime geçmeden önce kodun, ses yükleyicinin ve model döngüsünün sorunsuz çalıştığını 2 örnekle doğrulamak için:
```bash
!python src/train.py --smoke-test
```
*Bu komut sadece 2 eğitim ve 2 doğrulama kaydıyla 2 adımda tamamlanır (yaklaşık 15-30 saniye sürer).*

---

## 4. Whisper-Tiny Tam Fine-Tuning Eğitimi (T4 GPU)
T4 GPU üzerinde tam eğitimi başlatmak için:
```bash
!python src/train.py \
  --fp16 \
  --epochs 3 \
  --batch-size 8 \
  --grad-accum 2 \
  --learning-rate 1e-4 \
  --warmup-steps 50 \
  --output-dir results/checkpoints/E3
```

* **Protokol:**
  * Sadece `data/splits/train.tsv` (1.902 ses, ~8 saat) üzerinde eğitilir.
  * Checkpoint seçimi ve değerlendirme sadece `data/splits/validation.tsv` (210 ses, ~0.8 saat) ile yapılır.
  * En iyi validation WER sonucunu veren model otomatik olarak `results/checkpoints/E3/best/` altına kaydedilir.
  * Eğitim özeti ve donanım bilgileri `results/runs/E3_train.json` dosyasına yazılır.
  * `data/splits/test.tsv` test setine **asla dokunulmaz**.

---

## 5. Eğitilmiş Modeli (E3) Test Seti Üzerinde Değerlendirme
Eğitim bittikten sonra, en iyi checkpoint'i dondurulmuş 252 test kaydı üzerinde test edin:

```bash
# 1. E3 modeliyle test seti çıkarımı yapın
!python src/run_baseline.py --run E3

# 2. E1, E2 ve E3 modellerini ortak WER/CER metriği ile puanlayın
!python src/score_predictions.py --predictions \
  results/predictions/E1.tsv \
  results/predictions/E2.tsv \
  results/predictions/E3.tsv
```

Bu işlem sonucunda güncellenen `results/metrics.csv` tablosu ekrana yazdırılacaktır:
* **E1:** Vanilla Whisper-Tiny (Referans temel model)
* **E2:** MMS-1B Wav2Vec2-CTC (Harici karşılaştırıcı)
* **E3:** Fine-Tuned Whisper-Tiny (Grup katkısı modelimiz)

---

## 6. Sonuçları GitHub'a Kaydetme
Colab oturumu kapandığında dosyaların kaybolmaması için üretilen tahminleri, çalışma günlüklerini ve metrikleri deponuza commit edin:
```bash
!git status
!git add results/runs/E3.json results/runs/E3_train.json results/predictions/E3.tsv results/metrics.csv
!git commit -m "feat(T4): add fine-tuned Whisper-Tiny E3 training and test results"
!git push origin <dal-adı>
```
*(Model ağırlıklarını Hugging Face Hub'a veya Google Drive'a yedekleyebilirsiniz).*

