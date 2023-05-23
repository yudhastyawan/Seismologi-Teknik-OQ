# Seismologi-Teknik-OQ
Kumpulan script files untuk menjalankan HMTK Openquake dan pyGMT (atau Geopandas)

## Hal yang perlu di install
### Openquake
Openquake dapat diunduh di sini:
https://downloads.openquake.org/pkgs/windows/oq-engine/OpenQuake_Engine_3.11.5-1.exe

atau cek Openquake terbaru di link berikut ini:
https://downloads.openquake.org/pkgs/windows/oq-engine/

Install beberapa tambahan packages di openquake:

(Asumsi lokasi instalasi openquake di sini: `C:\Program Files\Openquake Engine\python3.6\`, 
jika tidak maka ubah lokasi tersebut sesuai dengan komputer masing-masing)

1. Buka Command Prompt
2. Install `geojson`: `"C:\Program Files\Openquake Engine\python3.6\python.exe" -m pip install geojson`
3. Install `jupyter`: `"C:\Program Files\Openquake Engine\python3.6\python.exe" -m pip install jupyter`
4. Daftarkan ke dalam kernel (opsional): `"C:\Program Files\Openquake Engine\python3.6\python.exe" -m ipykernel install --name "openquake" --display-name "openquake"`


Untuk uji coba, jalankan salah satu baris di bawah ini:

`"C:\Program Files\Openquake Engine\python3.6\Scripts\jupyter.exe" notebook`

atau

`"C:\Program Files\Openquake Engine\python3.6\Scripts\jupyter-notebook.exe"`

atau

`"C:\Program Files\Openquake Engine\python3.6\python.exe" -m notebook`

### Miniconda
Miniconda dapat diunduh di sini:
https://docs.conda.io/en/latest/miniconda.html

### PyGMT
1. Buka Anaconda Prompt
2. Buat environment pyGMT: `conda create --name pygmt --channel conda-forge pygmt`
3. Aktifkan environment `maps`: `conda activate pygmt`
4. Install `ipykernel`: `pip install ipykernel`
5. Daftarkan ke dalam kernel: `python -m ipykernel install --name "gmt" --display-name "pyGMT"`

### Geopandas
1. Buka Anaconda Prompt
2. Buat environment baru: `conda create -n maps python=3.9`
3. Aktifkan environment `maps`: `conda activate maps`
4. Install `geopandas`: `pip install geopandas`
5. Install `matplotlib`: `pip install matplotlib`
6. Install `ipykernel`: `pip install ipykernel`
7. Daftarkan ke dalam kernel: `python -m ipykernel install --name "maps" --display-name "maps"`

Data yang disimpan dalam format shp saat menggunakan Geopandas dapat digunakan di **QGIS**

## Cara menggunakan files yang ada di github ini
1. Download files di github ini dengan cara tekan tombol **Code -> Download ZIP**.
2. Ekstrak filenya dan tempatkan folder hasil ekstrak di tempat yang diinginkan.
3. Copy direktori/path folder tersebut, contoh: `C:\Users\USERNAME\Documents\Seismologi-Teknik-OQ-main`
4. Buka Command Prompt
5. Masuk ke direktori tersebut di Command Prompt: `cd C:\Users\USERNAME\Documents\Seismologi-Teknik-OQ-main`
6. Lalu jalankan jupyter notebook: `"C:\Program Files\Openquake Engine\python3.6\Scripts\jupyter.exe" notebook`
7. template notebook dapat menggunakan: `Template_openquake.ipynb` dan `Template_maps.ipynb` tergantung spesifik kernel

## Notebook sebelum revisi
Notebook sebelum direvisi dapat dilihat di folder **old**.

## Notebook vs. Kernel
| Notebook             | Kernel    |
|----------------------|-----------|
| OQ001_mengestimasi_area_katalog.ipynb        | maps |
| OQ002_declustering.ipynb        | openquake |
| OQ003_geometri_sumber_gempa.ipynb        | maps |
| OQ004_pemisahan_katalog_sumber_gempa.ipynb        | openquake |
| OQ005_visualisasi.ipynb        | maps |
| OQ006_menghindari_double_counting.ipynb        | openquake |
| OQ007_simpan_katalog_ke_shp.ipynb        | maps |
| OQ008_visualisasi.ipynb        | maps |
| OQ009_a_b_value.ipynb        | openquake |
| OQ010_persiapan_file_SHERIFS.ipynb        | maps |
| OQ011_jalankan_SHERIFS.ipynb        | openquake |
| OQ012_membuat_source_model_xml.ipynb        | openquake |
| OQ013_menjalankan_PSHA.ipynb        | none |
| OQ_Opt_001_membuat_source_model_xml_untuk_SHERIFS.ipynb        | openquake |
| OQ_Opt_002_menggabungkan_hazard_curves_dari_berbagai_sumber.ipynb        | openquake |
| old/deprecated_OQ-processing        | openquake |
| old/deprecated_OQ-visuals-geopandas | maps      |
| old/deprecated_OQ-visual-pygmt      | pyGMT     |
| old/OQ-processing        | openquake |
| old/OQ-visuals           | pyGMT     |
