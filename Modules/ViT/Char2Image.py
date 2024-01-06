from PIL import Image, ImageDraw, ImageFont
import numpy as np
from matplotlib import pyplot as plt
from fastNLP import logger


class Char2Image:
    def __init__(self, ttf_paths, font_size):
        self.font_paths = ttf_paths
        self.font_size = font_size
        self.image_size = (self.font_size, self.font_size)
        # 加载TrueType字体
        self.fonts = [ImageFont.truetype(font_path, self.font_size) for font_path in self.font_paths]
        logger.info(f'字体加载完成, 字体数量: {len(self.fonts)}')

    def get_image(self, word, font):
        # 创建一个空白图片
        img = Image.new('L', self.image_size, color='black')

        # 获取绘图上下文
        draw = ImageDraw.Draw(img)

        # 获取字符的宽度和高度
        bbox = draw.textbbox((0, 0), word, font=font)
        text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]

        # 计算字符在图片中的位置
        x = (self.image_size[0] - text_width) // 2
        y = (self.image_size[1] - text_height) // 2

        # 在图片上绘制字符
        draw.text((x, y), word, font=font, fill='white')

        # 获取灰度数据
        image_data = np.array(img)

        return image_data/255

    def get_images(self, words):
        images_datas = [[self.get_image(word, font) for font in self.fonts] for word in words]
        images_data = np.stack(images_datas)
        return images_data


if __name__ == '__main__':
    ttf_paths = ['data/simhei.ttf', 'data/xiaozhuan.ttf', 'data/trahei.ttf']
    font_size = 48
    char2img = Char2Image(ttf_paths, font_size)
    imgs = char2img.get_images('体')
    for img in imgs:
        plt.imshow(np.array(img))
        plt.show()
