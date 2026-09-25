# Kosar - The world's first hedge fund programmed 100% by AI

Project Name: Kosar
Version: 50
Meaning: In my culture, Kosar means tremendous and lasting prosperity and blessings.

What It Contains:
•	35,000 lines of code, including 15,000 lines of tests.
•	150 passing tests.
•	Written entirely with AI. I do not know how to write even a single line of code.

How the Project Took Shape and Evolved
v1 to v30:
From version 1 to version 30, I wrote the code using a combination of ChatGPT 4.5 and Gemini 2. Gemini was a weak programmer. As the codebase grew, it would summarize or delete parts of the existing code to optimize its output and manage its context window.
I searched the internet for a solution and discovered that programmers call long stretches of code crammed into a single file, such as Main.py, “spaghetti code.” The solution was to make the code modular: for example, splitting 1,000 lines of code into seven files, each with a different responsibility. This way, the model would only edit the file relevant to the prompt. This significantly improved the quality of the code, but it did not solve everything.
After a while, I realized that Gemini was a stubborn, rebellious model that paid little attention to my instructions. Initially, Gemini wrote the code, while ChatGPT acted as the critic. I noticed that when ChatGPT told Gemini to implement something in a particular way to prevent a potential problem X down the road, Gemini would ignore the advice and write it however it wanted. This became a blocker, causing numerous bugs and wasting a great deal of time.
So I switched their roles. I put ChatGPT in charge of writing the code and made Gemini the critic. The code improved considerably for two reasons: first, ChatGPT was inherently a better programmer; second, it was more up to date and searched the web more often.
Some time later, I saw promotions for Codex inside ChatGPT and installed it. I had previously researched IDEs with built-in AI agents. I always used Gemini’s Deep Research feature for my research, asking it to examine Reddit community feedback to find out how satisfied people were with a particular website or service. I would ask it to create a table showing the level of satisfaction with each product.
Research that took perhaps no more than 30 minutes saved me a great deal of money and time. Most of the time, its findings were accurate, too.
I had previously researched Cursor and its competitor this way and found widespread dissatisfaction with them, so I had decided against using them. But when I saw the promotions for Codex, I installed it because I had already used ChatGPT and been satisfied with it, and because I knew the company took a more economical approach.
The quality of the code improved yet again. Codex could now edit just the individual lines that needed changing, instead of rewriting the entire file.

Identifying the Problem and Addressing Its Root Cause
I had reached version 30, and the code was running in paper-trading mode on a server when I noticed that its performance was flat. It was neither making money nor losing it. I gave the code to Gemini Deep Research and asked it to critique it with complete honesty and absolutely no mercy. I asked it to compare the code with the architectures used by today’s major hedge funds, give it a score out of 20, and identify what I needed to add or remove to make it profitable.

The Results Came In
Gemini Deep Research gave my code a score of 4 out of 20. It was shocking, but it did not scare me. The biggest flaw was that I had put LLMs in charge of trading decisions. Gemini told me that major hedge funds use machine learning to predict market movements, while LLMs serve only to analyze sentiment.
Its other recommendations included adding meta-labeling, with XGBoost as the primary model and CatBoost as the secondary model; using the triple barrier method for model training; incorporating fractional differentiation into the features; adding an HMM to detect market regimes; adding OFI and VPIN; introducing a calibration layer; and several other improvements.

A Slap in the Face
Gemini’s report was asking me to smash all my idols, like the Prophet Abraham. Things that brought me neither benefit nor harm. A pile of stones and sticks I had gathered around myself. All the time I had wasted getting to version 30.
I had built an AI Senate consisting of 12 different AI systems, each independently analyzing the market. They voted. They held debates and tried to persuade one another by presenting arguments for and against. This happened in three stages: first, independent opinions; then, arguments from those in favor, those opposed, and those abstaining; and finally, another round of voting.
Now Gemini was telling me to throw away all this magnificent code, because magnificence was not going to pay my bills.
Its most devastating sentence was:
“Debates between LLMs in the AI Senate are merely an exchange of two hallucinations.”
It was harsh criticism, but I did not feel sad. I immediately started making changes, because the criticism was valid.
Machine learning learned from the past: if conditions Y had caused the price to rise X times, there was a probability Z that it would happen again. An LLM, however, was merely predicting the next sentences. Like a fortune-teller. With nothing to back it up.
So I decided to set LLMs aside for the time being, until I could use them where they belonged and where their strengths lay: sentiment analysis.
(This is why you can still see remnants of that system in the code on GitHub, especially in Senate.py.)

Beginning to Smash the Idols
After temporarily retiring the LLMs, I decided to implement the items on Gemini’s hedge fund architecture checklist one by one. I started with HMM.
By then, ChatGPT had taken on the role of writing prompts, while Codex received those prompts and wrote the code. After a while, I began to feel that the two of them were not up to the task. The codebase grew far beyond what seemed reasonable, and I felt that something was wrong.
Building a Factory for Bug-Free Code
During a casual conversation with a friend, I happened to learn that he was happier with Anthropic’s models than with OpenAI’s. I decided to try them, and the results were outstanding.
Fable was exactly what I had been missing. It followed my instructions precisely, without resisting or deviating from what I wanted. It had a strong memory and maintained its focus. If I mentioned something at the beginning of a conversation, it would not only remember it but also remind me later when the time came to act on it.
Within a short time, I realized that this model could help me write even the most complex code without bugs.
Together, We Know Everything
There is a very well-known saying in my country: “Together, we know everything.” It means that no matter how intelligent or knowledgeable a person is, they can still make mistakes and need to seek advice from others.
I knew that LLMs were modeled on the human brain, so they were not perfect either. That is why I built a coding pipeline: Fable as the system architect, ChatGPT as a critic focused on code review, and Opus as a critic focused on financial matters, numbers, and accounting.
The key was to assign each model the right role based on its natural strengths, maximizing productivity and minimizing errors. I had already learned part of this through working with ChatGPT. For models I had no experience with, such as Fable and Opus, I used Gemini Deep Research to investigate what each was best suited for.
Another strength of this architecture was the “difference in focus.” Each model examined the code with a different focus, allowing it to notice things the others might miss.
It is like hanging a picture on a wall. The person hanging it needs someone standing farther away to tell them whether it is crooked or straight. One person focuses on mounting the picture; the other focuses on whether it is level.
This is similar to the difference in focus between the primary model and the meta-model.
The Pipeline Structure That Worked
Fable wrote the initial prompt. I gave it to ChatGPT and Opus for review, then brought both of their assessments back to Fable.
Fable agreed with some points, disagreed with others, and partially agreed with the rest. For example, it might say, “The diagnosis is correct, but the proposed solution is wrong.”
I then sent Fable’s response back to the other two AIs so they could critique its decisions again. Sometimes this cycle repeated seven times, because each revision of the prompt introduced new problems.
Once the issues were resolved and I felt the prompt was ready, I gave it to Codex to write the code. I then sent Codex’s code to the AIs so they could review the results as well. Usually, only minor adjustments were needed, if any.
(I know that, as you read this, you are probably thinking that LLMs are designed to agree with one another. But this was different.)
Another important observation was that as the chat’s context window filled up, the model’s performance would begin to deteriorate. To address this, after several messages, I would ask it to create a Markdown handoff file documenting everything we had done and everything we planned to do next. I would then give it that file in a new chat to establish the context, and we would continue from there.
v31 to v50
Using this structure, the code was written to the highest standard possible. Everything Gemini had recommended was implemented. Each stage included walk-forward testing and its own dedicated tests. Telemetry, security considerations such as transactional logging—everything was in place.
It was time to test the system’s profitability before integrating the various components and going live.
The results showed remarkably strong—and surprisingly large—improvements. The model had managed to reduce maximum drawdown by as much as 81%. This far exceeded the industry benchmark outlined in Gemini’s roadmap. Gemini had stated that meta-models, particularly CatBoost, had achieved reductions in maximum drawdown of up to 30%.
That reminded me of the first rule of algorithmic trading:
If the results look exceptionally good, question them first. Celebrate later.
So I began testing.

Building the Ablation 81 Testing Lab
I wrote 15,000 lines of code to test the system from different angles: to determine whether it was generating alpha, which features were genuinely useful, and which were merely noise.
The tests returned an AUC of 0.51—almost equivalent to a coin toss—and delivered this verdict:
“Meta-model rejected.”
The testing lab found that 60% of the meta-model’s impressive results came down to luck, and only 40% to skill. When I saw the findings, I closed my laptop and walked out of the house crying. All my time and money had gone to waste.
Identifying the Root Cause
I stopped working on the project for a while and decided to focus on other things. Eventually, I remembered something strange I had noticed in the findings: a contradiction worth investigating.
In the folds covering periods before 2026, the primary model had been slightly profitable. But as soon as it entered the 2026 market, it began suffering heavy losses. It was in this 2026 market that the meta-model had identified a large proportion of the primary model’s incorrect signals and reduced maximum drawdown by as much as 81%.
That sudden deterioration raised a question: why had a profitable model abruptly started losing so much money?
I remembered something I always say: a problem must be solved at its source.
My suspicion shifted from the meta-model to the primary model. I examined its training setup—specifically, the triple barrier method configuration—and compared it with market behavior in 2026.
The configuration used large profit targets and long time horizons: a maximum holding period of 96 hours, a take-profit multiplier of 4, and a stop-loss multiplier of 2.
Comparing market behavior before and during 2026, I noticed that the earlier market had been predominantly bullish, offering substantial gains. This had allowed the primary model to produce positive results with that particular configuration.
But when the market began moving sideways in 2026 and lost its upward trend, the price would never reach the 4× take-profit barrier. Instead, it would hit the stop-loss barrier before the 96-hour time limit was reached.
I realized that the problem was not the meta-model. It was the primary model.
The primary model had been feeding the secondary model noise instead of signals. Because the meta-model could not find a pattern in that noise, it could not improve the results; it simply participated in fewer trades. Avoiding those trades prevented substantial losses, but that did not mean the meta-model was intelligent.
Now that the market no longer had that steady upward trend, I needed to change the triple barrier training configuration to match the behavior of the 2026 market, using smaller profit targets and shorter time horizons.
This reminded me of something from my earlier business research: adapting the business model of a successful merchant who lived 1,400 years ago.

The “Ibn 'Awf” Model
Abd al-Rahman Ibn 'Awf was a very wealthy merchant born in 580 CE in Mecca, in what is now Saudi Arabia. He was among the earliest converts to Islam and, under pressure from its opponents and enemies, migrated from Mecca to Medina with the Prophet Muhammad.
During this forced migration, he had to leave all his possessions and property behind and arrived in the new city in absolute poverty. Yet within 11 years, he became wealthy again—even wealthier than before, and this time, his wealth lasted.
What Does This Have to Do with Trading?
There are several connections. Ibn 'Awf was a traditional merchant. He did exactly what traders do today, just without the internet.
But my main reason for studying his business model was that he managed to become wealthy twice. To me, this meant that his success was not a matter of luck. He had learned the algorithm for getting rich.
What Was His Business Model?
Many people in this world are not fortunate enough to become wealthy. According to statistics, 99% of the world’s population will never accumulate $1 million in assets during their lifetime. So if someone manages to do it—and does it twice—their method is worth examining.
According to Ibn 'Awf himself, he started his business in the new city by selling oil and cheese from a small shop in the market. He explained that he added only a very small markup to his goods. His approach had two advantages.
First, his goods sold much faster than those of other shopkeepers, generating a high volume of sales. That volume ultimately produced exponential profits.
Second, he never sold on credit—a common practice in which customers received goods immediately and paid for them later, similar to today’s installment purchases. This meant he always had cash available, allowing him to buy new stock sooner and sell it sooner.
In essence, he built his considerable wealth by accelerating the sales cycle—similar to what Decathlon does in retail and what market makers do in online financial markets.
How Did I Implement His Business Model in My Software?
As I mentioned, I concluded that I needed to tighten the take-profit and stop-loss thresholds. Ibn 'Awf’s business model followed exactly the same principle: small profits and quick sales. That meant shortening the time horizons as well—for example, reducing the maximum holding period to 24 hours.
But I did not know the right configuration. In this industry, guesswork leads to lost capital. So I built another testing lab to evaluate dozens of configurations and identify which one would train the primary model effectively and produce a profit in the 2026 fold.
Eventually, I found a suitable configuration, and the system became profitable.
The Project’s Current Status
The project is on hold due to insufficient resources. Continuing to spend my time and money on it was no longer worthwhile. I chose to pursue more profitable work that requires much less expensive infrastructure to generate income.
This project had also taught me something particularly important: my long-held dream had come true. I no longer needed unreliable, overpriced programmers who delivered poor-quality work.
Anyone who has owned a business knows that its biggest obstacle—or its greatest driver of progress—is not advertising or regulations, but its employees. 
And, by the grace of God, I have finally been able to set that heavy burden down.
