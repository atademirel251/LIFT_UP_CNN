import os

# Silmek istediğiniz dosyanın yolu
file_path  = "C:/Users/atade/Desktop/1000verısetı/csv/desktop.ini"


# Dosyanın var olup olmadığını kontrol edin ve silin
if os.path.exists(file_path):
    os.remove(file_path)
    print(f"{file_path} dosyası silindi.")
else:
    print(f"{file_path} dosyası bulunamadı.")
