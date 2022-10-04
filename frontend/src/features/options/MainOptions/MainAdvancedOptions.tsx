import React, { ChangeEvent } from 'react';
import { RootState, useAppDispatch, useAppSelector } from '../../../app/store';
import { setShowAdvancedOptions } from '../optionsSlice';

export default function MainAdvancedOptions() {
  const showAdvancedOptions = useAppSelector(
    (state: RootState) => state.options.showAdvancedOptions
  );
  const dispatch = useAppDispatch();

  const handleShowAdvancedOptions = (e: ChangeEvent<HTMLInputElement>) =>
    dispatch(setShowAdvancedOptions(e.target.checked));

  return (
    <div className="advanced_options_checker">
      <input
        type="checkbox"
        name="advanced_options"
        id=""
        onChange={handleShowAdvancedOptions}
        checked={showAdvancedOptions}
      />
      <label htmlFor="advanced_options">Advanced Options</label>
    </div>
  );
}
