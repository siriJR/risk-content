#! /usr/bin/env python3

import re
import emoji
#import opencc
import jieba


class TextRegularization:
    def __init__(self, isZh=True,isCut=False):
        self.isZh = isZh
        self.isCut=isCut


    def strtr(self,text, replace):
        for s, r in replace.items():
            text = text.replace(s, r)
        return text

    def stopWordCase(self,text):
        """
        去掉停用词
        :param text:
        :return:
        """
        stopwords_dict = {
            '!': '',
            '.': '',
            ',': '',
            '#': '',
            '$': '',
            '%': '',
            '&': '',
            '*': '',
            '(': '',
            ')': '',
            '|': '',
            '?': '',
            '/': '',
            '@': '',
            '\'': '',
            '\'': '',
            ';': '',
            '[': '',
            ']': '',
            '{': '',
            '}': '',
            '+': '',
            '~': '',
            '-': '',
            '_': '',
            '=': '',
            '^': '',
            '<': '',
            '>': '',
            '　': '',
            '！': '',
            '。': '',
            '，': '',
            '￥': '',
            '（': '',
            '）': '',
            '？': '',
            '、': '',
            '“': '',
            '‘': '',
            '；': '',
            '【': '',
            '】': '',
            '——': '',
            '……': '',
            '《': '',
            '》': ''
        }
        return self.strtr(text, stopwords_dict)

    # def simplify(self,text):
    #     converter = opencc.OpenCC('t2s')
    #     return converter.convert(text)

    # def traditionalize(text):
    #     """ https://pypi.python.org/pypi/OpenCC
    #     """
    #     return opencc.convert(text, config='s2t.json')

    def sbcCase(self,text):
        """转换全角字符为半角字符
        SBC case to DBC case
        """
        return self.strtr(text, {
            "０": "0", "１": "1", "２": "2", "３": "3", "４": "4",
            "５": "5", "６": "6", "７": "7", "８": "8", "９": "9",
            'Ａ': 'A', 'Ｂ': 'B', 'Ｃ': 'C', 'Ｄ': 'D', 'Ｅ': 'E',
            'Ｆ': 'F', 'Ｇ': 'G', 'Ｈ': 'H', 'Ｉ': 'I', 'Ｊ': 'J',
            'Ｋ': 'K', 'Ｌ': 'L', 'Ｍ': 'M', 'Ｎ': 'N', 'Ｏ': 'O',
            'Ｐ': 'P', 'Ｑ': 'Q', 'Ｒ': 'R', 'Ｓ': 'S', 'Ｔ': 'T',
            'Ｕ': 'U', 'Ｖ': 'V', 'Ｗ': 'W', 'Ｘ': 'X', 'Ｙ': 'Y',
            'Ｚ': 'Z', 'ａ': 'a', 'ｂ': 'b', 'ｃ': 'c', 'ｄ': 'd',
            'ｅ': 'e', 'ｆ': 'f', 'ｇ': 'g', 'ｈ': 'h', 'ｉ': 'i',
            'ｊ': 'j', 'ｋ': 'k', 'ｌ': 'l', 'ｍ': 'm', 'ｎ': 'n',
            'ｏ': 'o', 'ｐ': 'p', 'ｑ': 'q', 'ｒ': 'r', 'ｓ': 's',
            'ｔ': 't', 'ｕ': 'u', 'ｖ': 'v', 'ｗ': 'w', 'ｘ': 'x',
            'ｙ': 'y', 'ｚ': 'z',
        })

    def circleCase(self,text):
        """转换①数字为半角数字
        ①②③④⑤⑥⑦⑧⑨⑩ case 全角
        ❹❸❽❼❽❼❾❽❼
        ➀➁➂➃➄➅➆➇➈
        """
        return self.strtr(text, {
            "➀": "1", "➁": "2", "➂": "3", "➃": "4", "➄": "5",
            "➅": "6", "➆": "7", "➇": "8", "➈": "9", "①": "1",
            "②": "2", "③": "3", "④": "4", "⑤": "5", "⑥": "6",
            "⑦": "7", "⑧": "8", "⑨": "9", "❶": "1", "❷": "2",
            "❸": "3", "❹": "4", "❺": "5", "❻": "6", "❼": "7",
            "❽": "8", "❾": "9", "➊": "1", "➋": "2", "➌": "3",
            "➍": "4", "➎": "5", "➏": "6", "➐": "7", "➑": "8",
            "➒": "9", 'Ⓐ': 'A', 'Ⓑ': 'B', 'Ⓒ': 'C', 'Ⓓ': 'D',
            'Ⓔ': 'E', 'Ⓕ': 'F', 'Ⓖ': 'G', 'Ⓗ': 'H', 'Ⓘ': 'I',
            'Ⓙ': 'J', 'Ⓚ': 'K', 'Ⓛ': 'L', 'Ⓜ': 'M', 'Ⓝ': 'N',
            'Ⓞ': 'O', 'Ⓟ': 'P', 'Ⓠ': 'Q', 'Ⓡ': 'R', 'Ⓢ': 'S',
            'Ⓣ': 'T', 'Ⓤ': 'U', 'Ⓥ': 'V', 'Ⓦ': 'W', 'Ⓧ': 'X',
            'Ⓨ': 'Y', 'Ⓩ': 'Z', 'ⓐ': 'a', 'ⓑ': 'b', 'ⓒ': 'c',
            'ⓓ': 'd', 'ⓔ': 'e', 'ⓕ': 'f', 'ⓖ': 'g', 'ⓗ': 'h',
            'ⓘ': 'i', 'ⓙ': 'j', 'ⓚ': 'k', 'ⓛ': 'l', 'ⓜ': 'm',
            'ⓝ': 'n', 'ⓞ': 'o', 'ⓟ': 'p', 'ⓠ': 'q', 'ⓡ': 'r',
            'ⓢ': 's', 'ⓣ': 't', 'ⓤ': 'u', 'ⓥ': 'v', 'ⓦ': 'w',
            'ⓧ': 'x', 'ⓨ': 'y', 'ⓩ': 'z',
            "㊀": "一", "㊁": "二", "㊂": "三", "㊃": "四",
            "㊄": "五", "㊅": "六", "㊆": "七", "㊇": "八",
            "㊈": "九",
        })

    def bracketCase(self,text):
        """转换⑴数字为半角数字
        ⑴ ⑵ ⑶ ⑷ ⑸ ⑹ ⑺ ⑻ ⑼case 全角
        """
        return self.strtr(text, {
            "⑴": "1", "⑵": "2", "⑶": "3", "⑷": "4", "⑸": "5",
            "⑹": "6", "⑺": "7", "⑻": "8", "⑼": "9",
            '🄐': 'A', '🄑': 'B', '🄒': 'C', '🄓': 'D', '🄔': 'E',
            '🄕': 'F', '🄖': 'G', '🄗': 'H', '🄘': 'I', '🄙': 'J',
            '🄚': 'K', '🄛': 'L', '🄜': 'M', '🄝': 'N', '🄞': 'O',
            '🄟': 'P', '🄠': 'Q', '🄡': 'R', '🄢': 'S', '🄣': 'T',
            '🄤': 'U', '🄥': 'V', '🄦': 'W', '🄧': 'X', '🄨': 'Y',
            '🄩': 'Z', '⒜': 'a', '⒝': 'b', '⒞': 'c', '⒟': 'd',
            '⒠': 'e', '⒡': 'f', '⒢': 'g', '⒣': 'h', '⒤': 'i',
            '⒥': 'j', '⒦': 'k', '⒧': 'l', '⒨': 'm', '⒩': 'n',
            '⒪': 'o', '⒫': 'p', '⒬': 'q', '⒭': 'r', '⒮': 's',
            '⒯': 't', '⒰': 'u', '⒱': 'v', '⒲': 'w', '⒳': 'x',
            '⒴': 'y', '⒵': 'z',
            "㈠": "一", "㈡": "二", "㈢": "三", "㈣": "四",
            "㈤": "五", "㈥": "六", "㈦": "七", "㈧": "八",
            "㈨": "九",
        })

    def dotCase(self,text):
        """转换⑴数字为半角数字
        ⒈⒉⒊⒋⒌⒍⒎⒏⒐case 全角
        """
        return self.strtr(text, {
            "⒈": "1",
            "⒉": "2",
            "⒊": "3",
            "⒋": "4",
            "⒌": "5",
            "⒍": "6",
            "⒎": "7",
            "⒏": "8",
            "⒐": "9",
        })

    def specialCase(self,text):
        """特殊字符比如希腊字符，西里尔字母
        """
        return self.strtr(text, {
            # 希腊字母
            "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H",
            "Ι": "I", "Κ": "K", "Μ": "M", "Ν": "N", "Ο": "O",
            "Ρ": "P", "Τ": "T", "Χ": "X", "α": "a", "β": "b",
            "γ": "y", "ι": "l", "κ": "k", "μ": "u", "ν": "v",
            "ο": "o", "ρ": "p", "τ": "t", "χ": "x",

            # 西里尔字母 (U+0400 - U+04FF)
            "Ѐ": "E", "Ё": "E", "Ѕ": "S", "І": "I", "Ї": "I",
            "Ј": "J", "Ќ": "K", "А": "A", "В": "B", "Е": "E",
            "З": "3", "Ζ": "Z", "И": "N", "М": "M", "Н": "H",
            "О": "O", "Р": "P", "С": "C", "Т": "T", "У": "y",
            "Х": "X", "Ь": "b", "Ъ": "b", "а": "a", "в": "B",
            "е": "e", "к": "K", "м": "M", "н": "H", "о": "O",
            "п": "n", "р": "P", "с": "c", "т": "T", "у": "y",
            "х": "x", "ш": "w", "ь": "b", "ѕ": "s", "і": "i",
            "ј": "j",

            "À": "A", "Á": "A", "Â": "A", "Ã": "A", "Ä": "A", "Å": "A", "Ā": "A", "Ă": "A", "Ă": "A",
            "Ç": "C", "Ć": "C", "Ĉ": "C", "Ċ": "C",
            "Ð": "D", "Ď": "D", "Đ": "D",
            "È": "E", "É": "E", "Ê": "E", "Ë": "E", "Ē": "E", "Ė": "E", "Ę": "E", "Ě": "E",
            "Ĝ": "G", "Ġ": "G", "Ģ": "G",
            "Ĥ": "H", "Ħ": "H",
            "Ì": "I", "Í": "I", "î": "I", "ï": "I", "į": "I",
            "Ĵ": "J",
            "Ķ": "K",
            "Ļ": "L", "Ł": "L",
            "Ñ": "N", "Ń": "N", "Ņ": "N", "Ň": "N",
            "Ò": "O", "Ó": "O", "Ô": "O", "Õ": "O", "Ö": "O", "Ő": "O",
            "Ŕ": "R", "Ř": "R",
            "Ś": "S", "Ŝ": "S", "Ş": "S", "Š": "S", "Ș": "S",
            "Ţ": "T", "Ť": "T", "Ț": "T",
            "Ù": "U", "Ú": "U", "Û": "U", "Ü": "U", "Ū": "U", "Ŭ": "U", "Ů": "U", "Ű": "U", "Ų": "U",
            "Ŵ": "W",
            "Ý": "Y", "Ŷ": "Y", "Ÿ": "Y",
            "Ź": "Z", "Ż": "Z", "Ž": "Z",

            "à": "a", "á": "a", "â": "a", "ã": "a", "ä": "a", "å": "a", "ā": "a", "ă": "a", "ą": "a",
            "ç": "c", "ć": "c", "ĉ": "c", "ċ": "c",
            "ď": "d", "đ": "d",
            "è": "e", "é": "e", "ê": "e", "ë": "e", "ē": "e", "ė": "e", "ę": "e", "ě": "e", "ə": "e",
            "ĝ": "g", "ġ": "g", "ģ": "g",
            "ĥ": "h", "ħ": "h",
            "ì": "i", "í": "i", "î": "i", "ï": "i", "ī": "i", "į": "i",
            "ĵ": "j",
            "ķ": "k",
            "ļ": "l",
            "ñ": "n", "ń": "n", "ņ": "n", "ň": "n",
            "ò": "o", "ó": "o", "ô": "o", "õ": "o", "ö": "o", "ő": "o", "ŕ": "r", "ř": "r",
            "ś": "s", "ŝ": "s", "ş": "s", "š": "s", "ș": "s",
            "ţ": "t", "ť": "t", "ț": "t",
            "ù": "u", "ú": "u", "û": "u", "ü": "u", "ū": "u", "ŭ": "u", "ů": "u", "ű": "u", "ų": "u",
            "ŵ": "w",
            "ý": "y", "ŷ": "y", "ÿ": "y",
            "ź": "z", "ż": "z", "ž": "z",

            # 罗马数字Roman numerals
            "Ⅰ": "I", "Ⅱ": "II", "Ⅲ": "III", "Ⅳ": "IV", "Ⅴ": "V", "Ⅵ": "VI", "Ⅶ": "VII",
            "Ⅷ": "VIII", "Ⅸ": "IX", "Ⅹ": "X", "Ⅺ": "XI", "Ⅻ": "XII", "Ⅼ": "L", "Ⅽ": "C",
            "Ⅾ": "D", "Ⅿ": "M",
            "ⅰ": "i", "ⅱ": "ii", "ⅲ": "iii", "ⅳ": "iv", "ⅴ": "v", "ⅵ": "vi", "ⅶ": "vii",
            "ⅷ": "viii", "ⅸ": "ix", "ⅹ": "x", "ⅺ": "xi", "ⅻ": "xii", "ⅼ": "l", "ⅽ": "c",
            "ⅾ": "d", "ⅿ": "m",
        })

    def emojiCase(self,text, language='zh'):
        """
        输入：🚀🌕
        输出：火箭：满月
        """
        return emoji.demojize(text, language=language)

    def spaceCase(self,text):
        #去空格
        regex = re.compile("\s+")
        text=re.sub(regex, " ", text.strip())
        #数字替换
        regex = re.compile("零|０|º|⁰|°|₀|０|⓪|0⃣️")
        text=re.sub(regex, "0",text)
        regex = re.compile("一|①|➀|⑴|⒈|❶|㈠|¹|₁|１|♳|⓵|I|1⃣️")
        text=re.sub(regex, "1",text)
        regex = re.compile("二|②|➁|⑵|⒉|❷|㈡|²|₂|２|♴|⓶|2⃣️")
        text=re.sub(regex, "2",text)
        regex = re.compile("三|③|➂|⑶|⒊|❸|㈢|³|₃|3|♵|⓷|3⃣️")
        text=re.sub(regex, "3",text)
        regex = re.compile("四|④|➃|⑷|⒋|❹|㈣|⁴|₄|４|♶|⓸|4⃣️")
        text=re.sub(regex, "4",text)
        regex = re.compile("五|⑤|➄|⑸|⒌|❺|㈤|⁵|₅|５|♷|⓹|5⃣️")
        text=re.sub(regex, "5",text)
        regex = re.compile("六|⑥|➅|⑹|⒍|❻|㈥|⁶|₆|6|♸|⓺|6⃣️")
        text=re.sub(regex, "6",text)
        regex = re.compile("七|⑦|➆|⑺|⒎|❼|㈦|⁷|₇|７|♹|⓻|7⃣️")
        text=re.sub(regex, "7",text)
        regex = re.compile("八|⑧|➇|⑻|⒏|❽|㈧|⁸|₈|８|⓼|8⃣️")
        text=re.sub(regex, "8",text)
        regex = re.compile("九|⑨|➈|⑼|⒐|❾|㈨|⁹|₉|９|⓽|9⃣️")
        text=re.sub(regex, "9",text)
        #网址替换
        regex = re.compile("((https?|ftp|news):\\/\\/)?([a-z0-9]([a-z0-9\\-]*[\\.。])+([a-z]{2}|aero|arpa|biz|com|coop|edu|gov|info|int|jobs|mil|museum|name|nato|net|org|pro|travel)|(([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\\.){3}([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5]))(\\/[a-z0-9_\\-\\.~]+)*(\\/([a-z0-9_\\-\\.]*)(\\?[a-z0-9+_\\-\\.%=&]*)?)?(#[a-z][a-z0-9_]*)?")
        text=re.sub(regex, "网址", text)
        #手机号替换
        regex = re.compile("(13[0-9]|14[01456879]|15[0-35-9]|16[2567]|17[0-8]|18[0-9]|19[0-35-9])\\d{8}")
        text=re.sub(regex, "手机号", text)
        #邮箱替换
        regex = re.compile("[*#\\u4e00-\\u9fa5 a-zA-Z0-9_.-]+@[a-zA-Z0-9-]+(\\.[a-zA-Z0-9-]+)*\\.[a-zA-Z0-9]{2,6}")
        text=re.sub(regex, "邮箱", text)
        #QQ替换
        regex = re.compile("[1-9]([0-9]{8,12})")
        patternFilter = r'(\d)\1{3,}|(?:01234|12345|23456|34567|45678|56789|98765|87654|76543|65432|54321|43210)'
        if re.search(patternFilter,text):
            tmp=""
        else:
            text=re.sub(regex, "QQ", text)
        #微信替换

        # regex = re.compile("[a-zA-Z][a-zA-Z\d_-]{5,19}")
        # if text.find("hhhh")>-1 or text.find("aaaa")>-1 or text.find("xxxx")>-1:
        #     tmp = ""
        # else:
        #     text=re.sub(regex, "微信", text)

        #regex = re.compile(r'[a-zA-Z](?!([a-zA-Z\d])\1{3,})[a-zA-Z\d_-]{5,19}')
        #regex = re.compile("[a-zA-Z][a-zA-Z\d_-]{5,19}")
        regex = re.compile(r'^[a-zA-Z](?!.*(.)\1{3})(?=.*[\d_-])[a-zA-Z\d_-]{5,19}$')
        text = re.sub(regex, "微信", text)

        return text

    def specialBizCase(self,text):
        return self.strtr(text, {
            "🅠":"q","ⓠ":"q","𝕢":"q","ᑫ":"q","𝐐":"q","ǫ":"q",
            "٩": "q","𝑄":"q","ℚ":"q","ꐎ":"q","kou":"q",
            "\\+":"加","➕":"加","add":"加","＋":"加","jia":"加",
            "mm":"妹妹","MM":"妹妹","镁铝":"美女","+":"加","yyds":"永远的神",
            "艹":"操","加v":"加微信","kp":"磕炮"

        })

    def extractWords(self,text):
        """抽出中文、英文、数字，忽略标点符号和特殊字符（表情）
        一般用与反垃圾
        """
        text = text.lower()  # 词向量只计算小写
        text = self.specialBizCase(text)  # 特殊业务字符
        text = self.stopWordCase(text)  # 去掉停用词
        #text = self.simplify(text)  # 繁体转简体--没必要
        text = self.sbcCase(text)  # 全角转半角
        text = self.circleCase(text)  # 特殊字符
        text = self.bracketCase(text)  # 特殊字符
        text = self.dotCase(text)  # 特殊字符
        text = self.specialCase(text)  # 特殊字符
        text = self.emojiCase(text)  # 颜表情转换
        text = self.spaceCase(text)  # 颜表情转换
        if self.isCut:
            if self.isZh:
                words = [item for item in jieba.cut(text, cut_all=False) if item not in [""," "]]
            else:
                words = text.strip().split()
        else:
            words=text
        return words


if __name__ == "__main__":
    inptStr="wxid_27636e7ytjoh12"
    textRe=TextRegularization()
    outputStr=textRe.extractWords(inptStr)
    print("输入：",inptStr,"\n")
    print("输出：", outputStr)




