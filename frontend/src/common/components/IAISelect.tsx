import { FormControl, FormLabel, Select, SelectProps } from '@chakra-ui/react';

interface Props extends SelectProps {
  label: string;
  styleClass?: string;
  validValues:
    | Array<number | string>
    | Array<{ key: string; value: string | number }>;
}
/**
 * Customized Chakra FormControl + Select multi-part component.
 */
const IAISelect = (props: Props) => {
  const {
    label,
    isDisabled,
    validValues,
    size = 'sm',
    fontSize = 'md',
    styleClass,
    ...rest
  } = props;
  return (
    <FormControl isDisabled={isDisabled} className={`iai-select ${styleClass}`}>
      <FormLabel
        fontSize={fontSize}
        marginBottom={1}
        flexGrow={2}
        whiteSpace="nowrap"
        className="iai-select-label"
      >
        {label}
      </FormLabel>
      <Select
        fontSize={fontSize}
        size={size}
        {...rest}
        className="iai-select-picker"
      >
        {validValues.map((opt) => {
          return typeof opt === 'string' || typeof opt === 'number' ? (
            <option key={opt} value={opt} className="iai-select-option">
              {opt}
            </option>
          ) : (
            <option key={opt.value} value={opt.value}>
              {opt.key}
            </option>
          );
        })}
      </Select>
    </FormControl>
  );
};

export default IAISelect;
