# Short Analysis of Model Results

## Example 1: Worcester Road

The first question asked which road name was displayed on the title card in the video. The correct answer was C, Worcester Road. Qwen base, fine-tuned Qwen, and VILA were all able to answer the question correctly. I think this example shows that all three models were able to understand text that appeared within the video and connect it to the multiple-choice options. Since this question depends on reading visible text, it also shows that the video and question were being processed correctly by each evaluation pipeline.

## Example 2: Premier Hand Car Wash

The second question asked which full service description appeared below the main title. The correct answer was A, Premier Hand Car Wash. Again, all three models answered correctly. I thought this question was useful because the model had to focus on the smaller text below the main title instead of only recognizing the most obvious text in the video. The fine-tuned Qwen model gave the same correct result as the base Qwen model and VILA.

## Example 3: Comparing the Models

For the third example, I compared the overall behavior of the three models because the provided demo test set only contained two evaluation questions. Qwen base, fine-tuned Qwen, and VILA each answered 2 out of 2 questions correctly, giving all three models an accuracy of 100%. Because the test set was so small, I do not think these results are enough to say that one model performed better than another or that fine-tuning improved Qwen. Instead, the results mainly show that the data conversion, LoRA fine-tuning, inference, answer parsing, scoring, and model comparison pipeline all worked successfully. A larger ReVA test set with more question types would be needed to make a stronger comparison between the models.
