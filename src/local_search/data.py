EDGES = {
    "E1": {
        "nama": "Koridor Timur",
        "a": "Asrama",
        "b": "Masjid",
        "kapasitas": 40,
        "waktu": 2,
    },
    "E2": {
        "nama": "Koridor Utara",
        "a": "Kelas",
        "b": "Masjid",
        "kapasitas": 30,
        "waktu": 2,
    },
    "E3": {
        "nama": "Koridor Barat",
        "a": "Perpustakaan",
        "b": "Ruang Makan",
        "kapasitas": 25,
        "waktu": 2,
    },
    "E4": {
        "nama": "Koridor Selatan",
        "a": "Lapangan",
        "b": "Asrama",
        "kapasitas": 35,
        "waktu": 2,
    },
    "E5": {
        "nama": "Koridor Tengah",
        "a": "Asrama",
        "b": "Kelas",
        "kapasitas": 20,
        "waktu": 2,
    },
    "E6": {
        "nama": "Aula Timur",
        "a": "Masjid",
        "b": "Ruang Makan",
        "kapasitas": 30,
        "waktu": 1,
    },
    "E7": {
        "nama": "Koridor Kampus",
        "a": "Perpustakaan",
        "b": "Kelas",
        "kapasitas": 20,
        "waktu": 3,
    },
    "E8": {
        "nama": "Koridor Lapangan",
        "a": "Lapangan",
        "b": "Kelas",
        "kapasitas": 25,
        "waktu": 2,
    },
}

GROUPS = [
    {
        "nama": "Kelompok A",
        "jumlah": 30,
        "asal": "Asrama",
        "tujuan": "Masjid",
        "deadline": 8,
        "alternatif": [["E1"], ["E5", "E2"]],
    },
    {
        "nama": "Kelompok B",
        "jumlah": 25,
        "asal": "Kelas",
        "tujuan": "Masjid",
        "deadline": 7,
        "alternatif": [["E2"], ["E5", "E1"]],
    },
    {
        "nama": "Kelompok C",
        "jumlah": 20,
        "asal": "Perpustakaan",
        "tujuan": "Ruang Makan",
        "deadline": 9,
        "alternatif": [["E3"], ["E7", "E2", "E6"]],
    },
    {
        "nama": "Kelompok D",
        "jumlah": 35,
        "asal": "Lapangan",
        "tujuan": "Asrama",
        "deadline": 10,
        "alternatif": [["E4"], ["E8", "E5"]],
    },
]

MAX_DELAY = 6
HARD_TOLERANCE = 6
WEIGHTS = {
    "capacity": 10,
    "conflict": 8,
    "late": 15,
    "waiting": 2,
    "distance": 1,
    "fairness": 2,
}
