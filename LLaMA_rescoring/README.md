## Candidate rescoring with LLaMA


First load your desired LLaMA version into Scratch. See [notes here](https://www.notion.so/reshef/ca8264e0872f49f58ab167f0aab9262c?v=45a6acd783114b0a9613110ae9b0a513&p=6099f87f50e8478e885907a23a6a3926&pm=s) for how to do this on Sherlock.

Then run `getTopKPredictions.py --evaluate_saved path/to/your/model` in the root repo directory. This will load an EMG transduction model and generate the top K sentences for each devset example (using beam search with 4-gram language model). 

After this finishes, a `.mat` file will be saved to the same folder as the model. If you're using the 7B-parameter LLaMA, you can use the Jupyter notebook to get performance after rescoring. Otherwise, you'll need to run `rescore_with_LargeLLaMA.py --mat_file path/to/mat/file` (NOT IMPLEMENTED).


