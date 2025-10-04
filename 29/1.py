import gc
import torch


def inference(sentences: list, custom_settings: dict):
    for sentence in sentences:
        try:
            with console.status("[bold bright_green] Model Inference..."):
                sentence_prompt = f'"{sentence}"是{custom_settings["class_list"]}里的什么类别？'

                # 使用 torch.no_grad() 减少内存使用
                with torch.no_grad():
                    response, history = model.chat(
                        tokenizer,
                        sentence_prompt,
                        history=custom_settings['pre_history'].copy()
                    )

            print(f'>>>[bold bright_red]sentence:{sentence}')
            print(f'>>>[bold bright_green]inference answer:{response}')
            print("*" * 80)

            # 清理内存
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"处理句子时出错: {sentence}, 错误: {e}")