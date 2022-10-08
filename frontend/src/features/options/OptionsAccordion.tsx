import { Accordion, ExpandedIndex } from '@chakra-ui/react';
import { RootState, useAppDispatch, useAppSelector } from '../../app/store';
import { setOpenAccordions } from '../system/systemSlice';
import InvokeAccordionItem, {
  InvokeAccordionItemProps,
} from './AccordionItems/InvokeAccordionItem';
import { ReactElement } from 'react';

type OptionsAccordionType = {
  [optionAccordionKey: string]: InvokeAccordionItemProps;
};

type OptionAccordionsType = {
  accordionInfo: OptionsAccordionType;
};

/**
 * Main container for generation and processing parameters.
 */
const OptionsAccordion = (props: OptionAccordionsType) => {
  const { accordionInfo } = props;

  const openAccordions = useAppSelector(
    (state: RootState) => state.system.openAccordions
  );

  const dispatch = useAppDispatch();

  /**
   * Stores accordion state in redux so preferred UI setup is retained.
   */
  const handleChangeAccordionState = (openAccordions: ExpandedIndex) =>
    dispatch(setOpenAccordions(openAccordions));

  const renderAccordions = () => {
    const accordionsToRender: ReactElement[] = [];
    if (accordionInfo) {
      Object.keys(accordionInfo).forEach((key) => {
        accordionsToRender.push(
          <InvokeAccordionItem
            key={key}
            header={accordionInfo[key as keyof typeof accordionInfo].header}
            feature={accordionInfo[key as keyof typeof accordionInfo].feature}
            options={accordionInfo[key as keyof typeof accordionInfo].options}
          />
        );
      });
    }
    return accordionsToRender;
  };

  return (
    <Accordion
      defaultIndex={openAccordions}
      allowMultiple
      reduceMotion
      onChange={handleChangeAccordionState}
      className="advanced-settings"
    >
      {renderAccordions()}
    </Accordion>
  );
};

export default OptionsAccordion;
