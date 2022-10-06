import {
  AccordionButton,
  AccordionIcon,
  AccordionItem,
  AccordionPanel,
} from '@chakra-ui/react';
import React, { ReactElement } from 'react';
import { Feature } from '../../../app/features';
import GuideIcon from '../../../common/components/GuideIcon';

export interface InvokeAccordionItemProps {
  header: ReactElement;
  feature: Feature;
  options: ReactElement;
}

export default function InvokeAccordionItem(props: InvokeAccordionItemProps) {
  const { header, feature, options } = props;

  return (
    <AccordionItem className="advanced-settings-item">
      <h2>
        <AccordionButton className="advanced-settings-header">
          {header}
          <GuideIcon feature={feature} />
          <AccordionIcon />
        </AccordionButton>
      </h2>
      <AccordionPanel className="advanced-settings-panel">
        {options}
      </AccordionPanel>
    </AccordionItem>
  );
}
