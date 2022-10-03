import {
  FormControl,
  NumberInput,
  NumberInputField,
  NumberIncrementStepper,
  NumberDecrementStepper,
  NumberInputProps,
  FormLabel,
} from '@chakra-ui/react';
import _ from 'lodash';
import { FocusEvent, useEffect, useState } from 'react';

const numberStringRegex = /^-?(0\.)?\.?$/;

interface Props extends Omit<NumberInputProps, 'onChange'> {
  styleClass?: string;
  label?: string;
  width?: string | number;
  showStepper?: boolean;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  clamp?: boolean;
  isInteger?: boolean;
}

/**
 * Customized Chakra FormControl + NumberInput multi-part component.
 */
const IAINumberInput = (props: Props) => {
  const {
    label,
    styleClass,
    isDisabled = false,
    showStepper = true,
    fontSize = '1rem',
    size = 'sm',
    width,
    textAlign,
    isInvalid,
    value,
    onChange,
    min,
    max,
    isInteger = true,
    ...rest
  } = props;

  /**
   * Using a controlled input with a value that accepts decimals needs special
   * handling. If the user starts to type in "1.5", by the time they press the
   * 5, the value has been parsed from "1." to "1" and they end up with "15".
   *
   * To resolve this, this component keeps a the value as a string internally,
   * and the UI component uses that. When a change is made, that string is parsed
   * as a number and given to the `onChange` function.
   */

  const [valueAsString, setValueAsString] = useState<string>(String(value));

  /**
   * When `value` changes (e.g. from a diff source than this component), we need
   * to update the internal `valueAsString`, but only if the actual value is different
   * from the current value.
   */
  useEffect(() => {
    if (!valueAsString.match(numberStringRegex) && value !== Number(valueAsString)) {
      setValueAsString(String(value));
    }
  }, [value, valueAsString]);

  const handleOnChange = (v: string) => {
    setValueAsString(v);
    // This allows negatives and decimals e.g. '-123', `.5`, `-0.2`, etc.
    if (!v.match(numberStringRegex)) {
      // Cast the value to number. Floor it if it should be an integer.
      onChange(isInteger ? Math.floor(Number(v)) : Number(v));
    }
  };

  /**
   * Clicking the steppers allows the value to go outside bounds; we need to
   * clamp it on blur and floor it if needed.
   */
  const handleBlur = (e: FocusEvent<HTMLInputElement>) => {
    const clamped = _.clamp(
      isInteger ? Math.floor(Number(e.target.value)) : Number(e.target.value),
      min,
      max
    );
    setValueAsString(String(clamped));
    onChange(clamped);
  };

  return (
    <FormControl
      isDisabled={isDisabled}
      isInvalid={isInvalid}
      className={`number-input ${styleClass}`}
    >
      {label && (
        <FormLabel
          fontSize={fontSize}
          marginBottom={1}
          flexGrow={2}
          whiteSpace="nowrap"
          className="number-input-label"
        >
          {label}
        </FormLabel>
      )}
      <NumberInput
        size={size}
        {...rest}
        className="number-input-field"
        value={valueAsString}
        keepWithinRange={true}
        clampValueOnBlur={false}
        onChange={handleOnChange}
        onBlur={handleBlur}
      >
        <NumberInputField
          fontSize={fontSize}
          className="number-input-entry"
          width={width}
          textAlign={textAlign}
        />
        <div
          className="number-input-stepper"
          style={showStepper ? { display: 'block' } : { display: 'none' }}
        >
          <NumberIncrementStepper className="number-input-stepper-button" />
          <NumberDecrementStepper className="number-input-stepper-button" />
        </div>
      </NumberInput>
    </FormControl>
  );
};

export default IAINumberInput;
