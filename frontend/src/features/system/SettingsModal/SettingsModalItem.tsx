import { FormControl, FormLabel, Switch } from '@chakra-ui/react';
import React from 'react';
import { useAppDispatch } from '../../../app/store';

export default function SettingsModalItem({
  settingTitle,
  isChecked,
  dispatcher,
}: {
  settingTitle: string;
  isChecked: boolean;
  dispatcher: any;
}) {
  const dispatch = useAppDispatch();
  return (
    <FormControl className="settings-modal-item">
      <FormLabel marginBottom={1}>{settingTitle}</FormLabel>
      <Switch
        isChecked={isChecked}
        onChange={(e) => dispatch(dispatcher(e.target.checked))}
      />
    </FormControl>
  );
}
