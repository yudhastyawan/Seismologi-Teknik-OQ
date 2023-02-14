# Seismologi-Teknik-OQ
Kumpulan script files untuk menjalankan HMTK Openquake dan pyGMT (atau Geopandas)

## Hal yang perlu di install
### Openquake
Openquake dapat diunduh di sini:
https://downloads.openquake.org/pkgs/windows/oq-engine/OpenQuake_Engine_3.11.5-1.exe

Install beberapa tambahan packages di openquake:

(Asumsi lokasi instalasi openquake di sini: `C:\Program Files\Openquake Engine\python3.6\`, 
jika tidak maka ubah lokasi tersebut sesuai dengan komputer masing-masing)

1. Buka Command Prompt
2. Install `utm`: `"C:\Program Files\Openquake Engine\python3.6\python.exe" -m pip install utm`
3. Install `jupyter`: `"C:\Program Files\Openquake Engine\python3.6\python.exe" -m pip install jupyter`


Untuk uji coba, jalankan salah satu baris di bawah ini:

`"C:\Program Files\Openquake Engine\python3.6\Scripts\jupyter.exe" notebook`

atau

`"C:\Program Files\Openquake Engine\python3.6\Scripts\jupyter-notebook.exe"`

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

## Cara menggunakan files yang ada di github ini
1. Download files di github ini dengan cara tekan tombol **Code -> Download ZIP**.
2. Ekstrak filenya dan tempatkan folder hasil ekstrak di tempat yang diinginkan.
3. Copy direktori/path folder tersebut, contoh: `C:\Users\USERNAME\Documents\Seismologi-Teknik-OQ-main`
4. Buka Command Prompt
5. Masuk ke direktori tersebut di Command Prompt: `cd C:\Users\USERNAME\Documents\Seismologi-Teknik-OQ-main`
6. Lalu jalankan jupyter notebook: `"C:\Program Files\Openquake Engine\python3.6\Scripts\jupyter.exe" notebook`
