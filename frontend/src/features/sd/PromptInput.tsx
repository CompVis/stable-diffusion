import { Textarea } from '@chakra-ui/react';
import { ChangeEvent } from 'react';
import { useAppDispatch, useAppSelector } from '../../app/hooks';
import { RootState } from '../../app/store';
import { setPrompt } from '../sd/sdSlice';

/**
 * Prompt input text area.
 */
const PromptInput = () => {
  const { prompt } = useAppSelector((state: RootState) => state.sd);
  const dispatch = useAppDispatch();

  const handleChangePrompt = (e: ChangeEvent<HTMLTextAreaElement>) =>
    dispatch(setPrompt(e.target.value));

  return (
    <Textarea
      id="prompt"
      name="prompt"
      resize="none"
      size={'lg'}
      height={'100%'}
      isInvalid={!prompt.length}
      onChange={handleChangePrompt}
      value={prompt}
      placeholder="I'm dreaming of..."
    />
  );
};

export default PromptInput;
