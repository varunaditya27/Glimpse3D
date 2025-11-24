import { motion, type HTMLMotionProps } from 'framer-motion';
import clsx from 'clsx';

interface ButtonProps extends Omit<HTMLMotionProps<"button">, "children"> {
    variant?: 'primary' | 'secondary' | 'outline';
    size?: 'sm' | 'md' | 'lg';
    isLoading?: boolean;
    children: React.ReactNode;
}

export const Button = ({
    children,
    className,
    variant = 'primary',
    size = 'md',
    isLoading,
    ...props
}: ButtonProps) => {
    return (
        <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            className={clsx(
                'btn',
                `btn--${variant}`,
                `btn--${size}`,
                className
            )}
            {...props}
        >
            {isLoading ? (
                <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin" style={{ width: '16px', height: '16px', border: '2px solid currentColor', borderTopColor: 'transparent', borderRadius: '50%', animation: 'spin 1s linear infinite' }} />
                    Processing...
                </span>
            ) : (
                children
            )}
        </motion.button>
    );
};
