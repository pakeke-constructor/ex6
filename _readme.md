


# ex6: Explicit LLM tool.

I was sick of all the LLM tools bloating my context window, and adding stuff implicitly that I didn't neccessarily want.  

I wanted a simpler, more explicit tool that allows for effective, nitty-gritty context engineering and workflow engineering.

This program, `ex6`, serves as a thin, simple alternative to claude-code, which does a lot less, and is entirely explicit.  
A lot of the UX/UI wi


## ex6 core tenets:
- No hidden context. Everything in the context window is explicit.
- High degree of customization/control.
- No leaky abstractions; serves as a thin, simple layer.

## features:
- less than 2000 loc pure python. No bloat.
- total customizability via "plugins."


## Plugins:
"Plugins" are just python files.  
"Installation?" Nope- just copy paste them into your `.ex6/` folder.  
All python files inside `.ex6` are loaded automatically.




# DISCLAIMER:
This tool is NOT meant to be used for non-technical folk.
If any of the below is true:
- You don't care about understanding the codebase
- You don't care about code-quality
- You just want a tool that works instantly
- You don't understand how LLMs actually work
Then this tool *probably* isn't for you



