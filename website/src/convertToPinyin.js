function convertToPinyin(base, tone) {

    // Mapping of base to accented forms
    const toneMap = {
        'a': ['ā', 'á', 'ǎ', 'à'],
        'e': ['ē', 'é', 'ě', 'è'],
        'i': ['ī', 'í', 'ǐ', 'ì'],
        'o': ['ō', 'ó', 'ǒ', 'ò'],
        'u': ['ū', 'ú', 'ǔ', 'ù'],
        'ü': ['ǖ', 'ǘ', 'ǚ', 'ǜ'],
    };

    // Find primary vowel in pinyin 
    const primaryVowel = (word) => {
        for (const vowel of ['a', 'o', 'e', 'i', 'u', 'ü']) {
            if (word.includes(vowel)) {
                return vowel;
            }
        }
        return null;
    };

    // Get vowel to place tone mark
    const vowel = primaryVowel(base);
    if (!vowel) {
        //console.error("No vowel found");
        return base;  
    }

    // Replace the base vowel with its accented version
    const toneIndex = tone - 1; 
    if (toneMap[vowel] && toneMap[vowel][toneIndex]) {
        return base.replace(vowel, toneMap[vowel][toneIndex]);
    } else {
        return base;
    }
}

// Example usage
// console.log(convertToPinyin('ma', 1)); // Outputs: mā
// console.log(convertToPinyin('ma', 2)); // Outputs: má
// console.log(convertToPinyin('ma', 3)); // Outputs: mǎ
// console.log(convertToPinyin('ma', 4)); // Outputs: mà
// console.log(convertToPinyin('ma', 0)); // Outputs: ma (neutral tone)
// console.log(convertToPinyin('ma', 5)); // Outputs: ma (neutral tone)

// Run test
// node src/convertToPinyin.js 

module.exports = convertToPinyin;
