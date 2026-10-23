Okay, so here's my thinking.

First, these recordings have kids' faces and voices in them, so nothing goes to any cloud AI. Everything runs on my own PC.

Second, I didn't want an AI just watching a class and saying "Good". That's a guess I can't explain. The computer measures real things: who talks and for how long, how fast kids answer, whether the tutor's face is on screen, whether the 8x8 background is there, the audio. A local AI model reads the transcript only to answer a fixed checklist (greeting, homework, closing) and has to show a real quote as proof. A simple weighted score, built from the QA team's own notes, turns that into a grade, and the "why" is written only from those facts.

I used Claude to plan and review, and Gemini to write code. My reviews caught a bug that could delete folders, a hidden shortcut in a test, a fallback that gave a weak class "Excellent", and frames with kids' faces left on disk. All fixed, and frames are now deleted after every run.

What worked: the logo check in the tutor's small corner video, checking whether the tutor is actually visible, and speed, about 3 minutes per hour of class.

What didn't, honestly: I only have 8 labelled classes. On those, the exact grade matched 0.75 of the time, within one grade 0.875. But I tuned the boundaries on those same classes, so that's optimistic. Fitted fairly it drops to 0.375, and guessing gets 0.25. The audio and warmth scores missed problems the reviewers flagged. Text checks only work in English; tested on Windows only.

So treat the grade as a rough signal, and the evidence as the real answer.
