# 🧠 Deep Neural Network - Universidad San Sebastián (2025)

**Profesor:** Sebastián Olivares  
**Correo:** sebastian.olivares@uss.cl  
**Repositorio del curso:** [https://github.com/solivare/DeepNeuralNetworkUSS](https://github.com/solivare/DeepNeuralNetworkUSS)

Este repositorio contiene todo el material del curso de Deep Neural Network impartido durante el primer semestre de 2025 en la Universidad San Sebastián. Aquí encontrarás ejemplos de clases, proyectos prácticos, y otros recursos didácticos.

---

## 📥 Clonar el repositorio

1. **¿Cómo crear tu cuenta de GitHub?**  

1. Ve a [https://github.com/](https://github.com/)
2. Haz clic en **Sign up**
3. Ingresa tu email y una contraseña segura
4. Elige un nombre de usuario (puedes usar tu nombre real o algo profesional)
5. Verifica tu cuenta por correo

2. **Crear un token de acceso personal en GitHub**  
   GitHub ya no permite autenticación con usuario/contraseña. Para clonar el repositorio privado o hacer `push`, necesitas un token.

   - Ve a [https://github.com/settings/tokens](https://github.com/settings/tokens)
   - Crea un token con permisos básicos para `repo` y `workflow`
   - Guarda el token (solo lo verás una vez)

3. **Clonar usando HTTPS**

```bash
git clone https://github.com/solivare/DeepNeuralNetworkUSS.git
```

> Si Git te pide usuario y contraseña, usa tu usuario de GitHub y el **token como contraseña**.

---

## 🌿 Crear tu propia rama (branch)

Cada estudiante debe trabajar en su propia rama. Para crearla:

```bash
cd DeepNeuralNetworkUSS
git checkout -b nombre_apellido
```

Por ejemplo:

```bash
git checkout -b juan_perez
```

Para subir tus cambios:

```bash
git add .
git commit -m "Mi primer commit"
git push origin nombre_apellido
```

---

## 🛠️ Comandos básicos de Git

| Acción                         | Comando                             |
|-------------------------------|-------------------------------------|
| Ver estado del repositorio    | `git status`                        |
| Cambiar de rama               | `git checkout nombre_rama`          |
| Traer cambios del repo        | `git pull origin main`              |
| Subir cambios a tu rama       | `git push origin nombre_rama`       |
| Ver ramas disponibles         | `git branch -a`                     |
| Fusionar ramas (merge)        | `git merge nombre_rama`             |

---

## 🤝 Contribución

- Cada estudiante debe trabajar en su propia rama
- Para sugerencias generales, usar el tab de *Issues* o comunicar directamente con el profesor
