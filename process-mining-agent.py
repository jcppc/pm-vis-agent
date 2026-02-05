# process_mining_agent.py
import os
import re
import warnings
import textwrap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from tqdm import tqdm
from IPython.display import display, Image as IPyImage
import openai
import anthropic
import matplotlib
matplotlib.use('Agg')  # garante backend não interativo
from matplotlib._pylab_helpers import Gcf


class ProcessMiningAgent:
    """
    Agente para geração e avaliação de código de mineração de processos
    """

    def __init__(self, generation_prompts=None, evaluation_prompts=None, api_keys=None):
        """
        Args:
            generation_prompts (dict): prompts para geração de código
            evaluation_prompts (dict): prompts para avaliação de código
            api_keys (dict): chaves de API para os diferentes provedores
        """
        self.generation_prompts = generation_prompts or {}
        self.evaluation_prompts = evaluation_prompts or {}
        self.api_keys = api_keys or {}

    def get_generation_prompt(self, prompt_name, question):
        """
        Retorna um prompt de geração formatado com a pergunta

        Args:
            prompt_name (str): Nome do prompt
            question (str): Pergunta a ser inserida no prompt

        Returns:
            str: Prompt formatado
        """
        if prompt_name not in self.generation_prompts:
            raise ValueError(f"Prompt '{prompt_name}' não encontrado. Prompts disponíveis: {list(self.generation_prompts.keys())}")

        return self.generation_prompts[prompt_name].format(question=question)

    def get_evaluation_prompt(self, prompt_name, question, generated_code, output_text):
        """
        Retorna um prompt de avaliação formatado

        Args:
            prompt_name (str): Nome do prompt
            question (str): Pergunta original
            generated_code (str): Código gerado
            output_text (str): Saída da execução do código

        Returns:
            str: Prompt formatado
        """
        if prompt_name not in self.evaluation_prompts:
            raise ValueError(f"Prompt de avaliação '{prompt_name}' não encontrado. Prompts disponíveis: {list(self.evaluation_prompts.keys())}")

        return self.evaluation_prompts[prompt_name].format(
            question=question,
            generated_code=generated_code,
            output_text=output_text
        )

    # ── Geração de código ──────────────────────────────────────────────

    def generate_code_openai(self, prompt, model):
        """
        Gera código usando a API da OpenAI
        """
        if 'openai' not in self.api_keys:
            raise ValueError("API key para OpenAI não fornecida. Use agent.api_keys['openai'] = 'sua-chave'")

        original_key = openai.api_key if hasattr(openai, 'api_key') else None
        openai.api_key = self.api_keys['openai']

        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_completion_tokens=4000,
            )
            return response['choices'][0]['message']['content']
        finally:
            if original_key is not None:
                openai.api_key = original_key

    def generate_code_claude(self, prompt, model):
        """
        Gera código usando Claude (Anthropic)
        """
        if 'claude' not in self.api_keys:
            raise ValueError("API key para Claude (Anthropic) não fornecida. Use agent.api_keys['claude'] = 'sua-chave'")

        client = anthropic.Anthropic(api_key=self.api_keys['claude'])
        response = client.messages.create(
            model=model,
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.content[0].text

    def generate_code_ollama(self, prompt, model):
        """
        Gera código usando Ollama (modelos locais)
        """
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': model,
                'prompt': prompt,
                'stream': False
            }
        )
        return response.json()['response']

    def generate_code_gemini(self, prompt, model):
        """
        Gera código usando a API do Google Gemini (via REST)
        """
        if 'google' not in self.api_keys:
            raise ValueError("API key para Google Gemini não fornecida. Use agent.api_keys['google'] = 'sua-chave'")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.api_keys['google']}"
        headers = {"Content-Type": "application/json"}
        data = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0}
        }

        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()

        return result['candidates'][0]['content']['parts'][0]['text']

    # ── Avaliação de código ────────────────────────────────────────────

    def evaluate_text_openai(self, question, generated_code, output_text, prompt_name, model):
        """
        Avalia a resposta usando a API da OpenAI
        """
        if 'openai' not in self.api_keys:
            raise ValueError("API key para OpenAI não fornecida. Use agent.api_keys['openai'] = 'sua-chave'")

        original_key = openai.api_key if hasattr(openai, 'api_key') else None
        openai.api_key = self.api_keys['openai']

        prompt = self.get_evaluation_prompt(prompt_name, question, generated_code, output_text)

        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            grade_str = response['choices'][0]['message']['content'].strip()
            match = re.search(r"\d+(?:\.\d+)?", grade_str)
            if not match:
                raise ValueError(f"Não foi possível encontrar um número na resposta de avaliação: '{grade_str}'")
            return float(match.group())
        finally:
            if original_key is not None:
                openai.api_key = original_key

    def evaluate_text_claude(self, question, generated_code, output_text, prompt_name, model):
        """
        Avalia a resposta usando a API do Claude (Anthropic)
        """
        if 'anthropic' not in self.api_keys:
            raise ValueError("API key para Anthropic não fornecida. Use agent.api_keys['anthropic'] = 'sua-chave'")

        client = anthropic.Anthropic(api_key=self.api_keys['anthropic'])

        prompt = self.get_evaluation_prompt(prompt_name, question, generated_code, output_text)

        response = client.messages.create(
            model=model,
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}]
        )
        grade_str = response.content[0].text.strip()
        match = re.search(r"\d+(?:\.\d+)?", grade_str)
        if not match:
            raise ValueError(f"Não foi possível encontrar um número na resposta de avaliação: '{grade_str}'")
        return float(match.group())

    def evaluate_text_ollama(self, question, generated_code, output_text, prompt_name, model):
        """
        Avalia a resposta usando Ollama (modelos locais)
        """
        prompt = self.get_evaluation_prompt(prompt_name, question, generated_code, output_text)

        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': model,
                'prompt': prompt,
                'stream': False
            }
        )
        grade_str = response.json()['response'].strip()
        match = re.search(r"\d+(?:\.\d+)?", grade_str)
        if not match:
            raise ValueError(f"Não foi possível encontrar um número na resposta de avaliação: '{grade_str}'")
        return float(match.group())

    def evaluate_text_gemini(self, question, generated_code, output_text, prompt_name, model):
        """
        Avalia a resposta usando a API do Google Gemini
        """
        if 'google' not in self.api_keys:
            raise ValueError("API key para Google Gemini não fornecida. Use agent.api_keys['google'] = 'sua-chave'")

        prompt = self.get_evaluation_prompt(prompt_name, question, generated_code, output_text)

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.api_keys['google']}"
        headers = {"Content-Type": "application/json"}
        data = {"contents": [{"parts": [{"text": prompt}]}]}

        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()

        grade_str = result['candidates'][0]['content']['parts'][0]['text'].strip()
        match = re.search(r"\d+(?:\.\d+)?", grade_str)
        if not match:
            raise ValueError(f"Não foi possível encontrar um número na resposta de avaliação: '{grade_str}'")
        return float(match.group())

    # ── Utilitários ────────────────────────────────────────────────────

    def get_model_function(self, model_type, task):
        """
        Retorna a função adequada para o tipo de modelo e tarefa

        Args:
            model_type (str): Tipo de modelo ('openai', 'claude', 'ollama' ou 'gemini')
            task (str): Tipo de tarefa ('generation' ou 'evaluation')

        Returns:
            function: Função correspondente ao modelo e tarefa
        """
        model_functions = {
            "generation": {
                "openai": self.generate_code_openai,
                "claude": self.generate_code_claude,
                "ollama": self.generate_code_ollama,
                "gemini": self.generate_code_gemini,
            },
            "evaluation": {
                "openai": self.evaluate_text_openai,
                "claude": self.evaluate_text_claude,
                "ollama": self.evaluate_text_ollama,
                "gemini": self.evaluate_text_gemini,
            }
        }

        return model_functions[task][model_type]

    def execute_code(self, code, log, df, output_folder):
        """
        Executa o código gerado em ambiente controlado e salva figuras diretamente na pasta de output.

        Args:
            code (str): Código a ser executado
            log: Log de eventos (pm4py)
            df: DataFrame pandas
            output_folder (str): Pasta onde os PNGs das figuras serão salvos

        Returns:
            tuple: (output, imgs, success)
        """
        os.makedirs(output_folder, exist_ok=True)

        # Registrar quais figuras já existiam antes da execução
        before_ids = {id(m.canvas.figure) for m in Gcf.get_all_fig_managers()}

        try:
            local_vars = {
                'log': log,
                'df': df.copy() if df is not None else None
            }
            blocks = re.findall(r"```python(.*?)```", code, re.DOTALL) or [code]
            for block in blocks:
                exec(textwrap.dedent(block), globals(), local_vars)
            output = local_vars.get('resultado', 'Sem resultado explícito')
            success = True
        except Exception as e:
            output = f"Erro na execução: {e}"
            success = False

        # Capturar as figuras novas e salvá-las
        imgs = []
        for mgr in Gcf.get_all_fig_managers():
            fig = mgr.canvas.figure
            if id(fig) not in before_ids:
                idx = len(imgs) + 1
                img_path = os.path.join(output_folder, f"figure_{idx}.png")
                fig.savefig(img_path, bbox_inches='tight')
                imgs.append(img_path)
                plt.close(fig)

        plt.close('all')
        return output, imgs, success

    def run_process_mining_batch(
        self,
        questions_df,
        output_root,
        gen_model_type,
        gen_model_name,
        eval_model_type,
        eval_model_name,
        prompt_name,
        eval_prompt_name,
        log,
        df,
        pause_between_questions=False
    ):
        """
        Executa um batch de processamento de mineração de processos

        Args:
            questions_df (DataFrame): DataFrame com coluna 'Question'
            output_root (str): Diretório raiz para salvar os resultados
            gen_model_type (str): 'openai', 'claude', 'ollama' ou 'gemini'
            gen_model_name (str): Nome do modelo de geração
            eval_model_type (str): 'openai', 'claude', 'ollama' ou 'gemini'
            eval_model_name (str): Nome do modelo de avaliação
            prompt_name (str): Chave no self.generation_prompts
            eval_prompt_name (str): Chave no self.evaluation_prompts
            log: Log de eventos (pm4py)
            df: DataFrame convertido do log
            pause_between_questions (bool): Pausa para revisão entre questões

        Returns:
            list: Resultados de cada questão
        """
        if prompt_name not in self.generation_prompts:
            raise ValueError(f"Prompt '{prompt_name}' não encontrado.")
        if eval_prompt_name not in self.evaluation_prompts:
            raise ValueError(f"Prompt de avaliação '{eval_prompt_name}' não encontrado.")
        if log is None or df is None:
            raise ValueError("Log e DataFrame são obrigatórios.")

        if gen_model_type == "openai" or eval_model_type == "openai":
            if 'openai' not in self.api_keys:
                raise ValueError("API key para OpenAI não fornecida")
        if gen_model_type == "claude" or eval_model_type == "claude":
            if 'claude' not in self.api_keys:
                raise ValueError("API key para claude não fornecida")

        os.makedirs(output_root, exist_ok=True)
        warnings.filterwarnings("ignore")
        plt.ioff()

        code_gen_fn = self.get_model_function(gen_model_type, "generation")
        eval_fn = self.get_model_function(eval_model_type, "evaluation")

        results = []
        for i, question in enumerate(tqdm(questions_df['Question'], desc="Processando"), start=1):
            folder = os.path.join(output_root, f"Question_{i}")
            os.makedirs(folder, exist_ok=True)

            # 1) Geração de código
            prompt = self.get_generation_prompt(prompt_name, question)
            codigo = code_gen_fn(prompt, model=gen_model_name)

            # 2) Execução do código e salvamento de figuras
            print(f"\nQuestão {i}: Executando código e salvando figuras em {folder}")
            output, imgs, success = self.execute_code(codigo, log, df, output_folder=folder)
            print("Resultado da execução:\n", output)

            # 3) Exibir figuras salvas no notebook
            if imgs:
                print("\nFiguras salvas:")
                for img in imgs:
                    print(" -", img)
                    display(IPyImage(filename=img))

            # 4) Avaliação
            nota = eval_fn(
                question=question,
                generated_code=codigo,
                output_text=output,
                prompt_name=eval_prompt_name,
                model=eval_model_name
            )
            print(f"\nNota atribuída: {nota}/10.0")

            # 5) Salvamento de artefatos de texto
            artifacts = {
                'prompt.txt': prompt,
                'codigo.txt': codigo,
                'resposta.txt': str(output),
                'nota_final.txt': str(nota),
                'modelo_geracao.txt': f"{gen_model_type}:{gen_model_name}",
                'modelo_avaliacao.txt': f"{eval_model_type}:{eval_model_name}",
                'prompt_geracao.txt': prompt_name,
                'prompt_avaliacao.txt': eval_prompt_name
            }
            for fn, text in artifacts.items():
                with open(os.path.join(folder, fn), 'w', encoding='utf-8') as f:
                    f.write(text)

            results.append({
                'question': question,
                'folder': folder,
                'nota_final': nota,
                'imagens': imgs,
                'modelo_geracao': f"{gen_model_type}:{gen_model_name}",
                'modelo_avaliacao': f"{eval_model_type}:{eval_model_name}",
                'prompt_geracao': prompt_name,
                'prompt_avaliacao': eval_prompt_name
            })

            if pause_between_questions:
                input("Pressione Enter para continuar...")

        # Sumário final
        print("\n=== RESUMO FINAL ===")
        for r in results:
            print(f"Q{i}: {r['question']} → Nota: {r['nota_final']} (pasta: {r['folder']})")
        return results