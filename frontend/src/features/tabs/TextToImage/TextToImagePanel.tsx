import React from 'react';
import { RootState, useAppSelector } from '../../../app/store';
import MainOptions from '../../options/MainOptions/MainOptions';
import OptionsAccordion from '../../options/OptionsAccordion';
import ProcessButtons from '../../options/ProcessButtons/ProcessButtons';
import PromptInput from '../../options/PromptInput/PromptInput';

export default function TextToImagePanel() {
  const showAdvancedOptions = useAppSelector(
    (state: RootState) => state.options.showAdvancedOptions
  );
  return (
    <div className="text-to-image-panel">
      <PromptInput />
      <ProcessButtons />
      <MainOptions />
      {showAdvancedOptions ? <OptionsAccordion /> : null}
    </div>
  );
}
