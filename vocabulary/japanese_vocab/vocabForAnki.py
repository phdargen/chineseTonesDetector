import csv
import pykakasi
from pykakasi import kakasi
import re

def is_kanji(char):
    return '\u4e00' <= char <= '\u9faf'

def get_furigana(kanji, kana):
    result = []
    kanji_start = 0
    kana_start = 0
    
    while kanji_start < len(kanji):
        # Find the next kanji character or sequence
        kanji_end = kanji_start
        while kanji_end < len(kanji) and is_kanji(kanji[kanji_end]):
            kanji_end += 1
        
        if kanji_start == kanji_end:
            # No kanji found, add the character as is
            result.append(kanji[kanji_start])
            kanji_start += 1
            kana_start += 1
        else:
            # Kanji sequence found, find corresponding kana
            kanji_seq = kanji[kanji_start:kanji_end]
            kana_end = kana_start + len(kanji_seq)
            while kana_end <= len(kana) and kana[kana_start:kana_end] not in kanji:
                kana_end += 1
            
            if kana_end <= len(kana):
                result.append(f"{kanji_seq}[{kana[kana_start:kana_end]}]")
                kanji_start = kanji_end
                kana_start = kana_end
            else:
                # If no match found, add kanji as is
                result.append(kanji_seq)
                kanji_start = kanji_end
                kana_start += len(kanji_seq)
    
    # Add any remaining kana
    if kana_start < len(kana):
        result.append(kana[kana_start:])
    
    return ''.join(result)

def read_and_write_vocabulary_file(input_file_path, output_file_path):
    fieldnames = ['Kana', 'Kanji', 'English', 'Genki Lesson', 'Furigana']

    with open(input_file_path, mode='r', encoding='utf-8') as infile, \
         open(output_file_path, mode='w', encoding='utf-8', newline='') as outfile:
        
        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            furigana_text = get_furigana(row['Kanji'], row['Kana'])
            
            writer.writerow({
                'Kana': row['Kana'],
                'Kanji': row['Kanji'],
                'English': row['English'],
                'Genki Lesson': row['Genki Lesson'],
                'Furigana': furigana_text
            })

# Test
test_cases = [
        ("新聞", "しんぶん"),
        ("食べる", "たべる"),
        ("お兄さん", "おにいさん"),
        ("日本語", "にほんご")
    ]
    
for kanji, kana in test_cases:
    furigana_result = get_furigana(kanji, kana)
    print(f"Input: {kanji} ({kana}) -> Output: {furigana_result}")

# Write output
input_file_path = 'genki.csv'  
output_file_path = 'genki_with_furigana.csv'  
read_and_write_vocabulary_file(input_file_path, output_file_path)
