import * as InvokeAI from '../../app/invokeai';

const promptToString = (prompt: InvokeAI.Prompt): string => {
  if (prompt.length === 1) {
    return prompt[0].prompt;
  }

  return prompt
    .map(
      (promptItem: InvokeAI.PromptItem): string =>
        `${promptItem.prompt}:${promptItem.weight}`
    )
    .join(' ');
};

export default promptToString;
