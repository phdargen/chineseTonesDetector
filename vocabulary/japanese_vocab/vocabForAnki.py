import csv
import pykakasi
from pykakasi import kakasi

def get_furigana(text):
    kks = kakasi()
    result = kks.convert(text)
    
    furigana_text = ""
    for item in result:
        if item['orig'] != item['hira']:
            furigana_text += f"{item['orig']}[{item['hira']}]"
        else:
            furigana_text += item['orig']
    
    return furigana_text

def read_and_write_vocabulary_file(input_file_path, output_file_path):
    with open(input_file_path, mode='r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        
        # Define output fieldnames including furigana
        fieldnames = ['Kana', 'Kanji', 'English', 'Genki Lesson', 'Furigana']
        
        with open(output_file_path, mode='w', encoding='utf-8', newline='') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for row in reader:
                kana = row['Kana']
                kanji = row['Kanji']
                english = row['English']
                genki_lesson = row['Genki Lesson']
                
                # Generate furigana in anki format: kanji[hiragana]
                furigana_text = get_furigana(kanji)
                
                # Write output 
                writer.writerow({
                    'Kana': kana,
                    'Kanji': kanji,
                    'English': english,
                    'Genki Lesson': genki_lesson,
                    'Furigana': furigana_text
                })

# Test
japanese_text = "お兄さん"
furigana_result = get_furigana(japanese_text)
print(furigana_result)

# Write output
input_file_path = 'genki.csv'  
output_file_path = 'genki_with_furigana.csv'  
read_and_write_vocabulary_file(input_file_path, output_file_path)
