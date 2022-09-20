import { Textarea } from '@chakra-ui/react';
import {
  ChangeEvent,
  KeyboardEvent,
} from 'react';
import { useAppDispatch, useAppSelector } from '../../app/store';
import { generateImage } from '../../app/socketio/actions';
import { RootState } from '../../app/store';
import { setPrompt } from '../options/optionsSlice';

/**
 * Prompt input text area.
 */
const PromptInput = () => {
  const { prompt } = useAppSelector((state: RootState) => state.options);
  const dispatch = useAppDispatch();

  const handleChangePrompt = (e: ChangeEvent<HTMLTextAreaElement>) =>
    dispatch(setPrompt(e.target.value));

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && e.shiftKey === false) {
      e.preventDefault();
      dispatch(generateImage())
    }
  };

  return (
    <Textarea
      id="prompt"
      name="prompt"
      resize="none"
      size={'lg'}
      height={'100%'}
      isInvalid={!prompt.length}
      onChange={handleChangePrompt}
      onKeyDown={handleKeyDown}
      value={prompt}
      placeholder="I'm dreaming of..."
    />
  );
};

export default PromptInput;
