# twitchbotsuperalpha
Supertwitchbotalpha
 SuperDuperTwitchBot
    Copyright (C) <2023>  <Travis Nan tnan2>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

******************************************************************************************************
I've made 90-95% of this code generate through gpt3 on chat.openai.com

I had 0 coding experience with Python prior to April 1st, 2023. Now I feel semi-competent on April 3rd, 2023. Insane.

It has taught me an insane amount about python, but I'm not exactly a genius. 

I'm hosting this code publicly, feel free to edit and tweak please just give
credit where it is due, which is again 90-95% to the creators of gpt3.

Please let me know if you're able to get this to work, as it's going to be
a project of mine over the next while unless I find other ways or scripts 
that can implement the functionalities I am seeking for this twitchbot.

I am also using the readme to keep track of the featurelist I have at the time of release to the repository.

Within the feature list currently is:

-Connects to Twitch chat using TwitchIO library
-Uses Hugging Face's ChatGPT to imitate the streamer's speaking style for generating chat messages
-Automatically types messages in the chat that imitate the streamer's speaking style
-Sends TTS messages in the streamer's speaking style to thank new subscribers using the Speechify API
-Randomizes response delay to avoid spamming the chat
-Responds to individual users within the chat, while responding to subscribers more frequently than non-subscribers,
 and also talking to the chat as a whole
-Includes moderator controls for clearing messages, toggling TTS, and other features
-Includes a logging system to keep track of important events and errors
-Displays a customizable CSS interface (Flask Interface)
-Automatic thank you message for new followers that imitates the streamer's speaking style
-Automatic thank you message for new subscribers that imitates the streamer's speaking style
-TTS (Text-to-Speech) message for new subscribers that imitates the streamer's speaking style
-TTS (Text-to-Speech) message for resubscribers that imitates the streamer's speaking style
-Moderator Management and Controls
-Integration with a language model for generating chat messages that imitates the streamer's speaking style
-Uses a random voice from Speechify that imitates the streamer's speaking style when a user cheers over 1000 bits
-Gambling/Point System with various games
-Provides customizable controls for every feature of the bot

Honestly, probably a few other features I have forgotten. Many features are just awaiting API / bot / auth setup.

Each function has been listed with its purpose within the .py code, as well as the editable .txt document which is
a copy of the current superalpha.py file. 

Thanks for your time, and good luck if you choose to help finish this code.

*****************************************************************************************************
