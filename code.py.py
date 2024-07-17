import click
from transformers import GPT3Tokenizer, GPT3ForSequenceClassification

# Set up Hugging Face Transformers
tokenizer = GPT3Tokenizer.from_pretrained("gpt-3-davinci-002")
model = GPT3ForSequenceClassification.from_pretrained("gpt-3-davinci-002")

@click.command()
@click.option('-t', '--text-file', type=click.File('r'), help='Text file to summarize')
@click.option('-x', '--text', type=str, help='Text to summarize')
def summarize(text_file, text):
    try:
        if text_file:
            text = text_file.read()
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model(**inputs)
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        click.echo(f'Summary: {summary}')
    except Exception as e:
        click.echo(f'Error: {str(e)}')

if __name__ == '__main__':
    summarize()